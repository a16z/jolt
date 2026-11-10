//! Proof-scoped native operands for the paired Product, ICR and Input slots.
mod terms;
pub(crate) use terms::InputTerms;

use super::instruction_input::InstructionInputRow;
use crate::KernelError;
use bytemuck_derive::{Pod, Zeroable};
use jolt_akita::AkitaField as F;
use jolt_claims::protocols::jolt::JoltPolynomialId;
use jolt_field::{Ring, Zero};
use jolt_metal::runtime::{Device, DeviceBuffer};
use jolt_metal::{ErrorClass, MetalError};
use jolt_riscv::{CircuitFlags, InstructionFlags, JoltTraceRow};
use jolt_sumcheck::SumcheckError;
use jolt_witness::witnesses::{lookup_values, Extract, NextIsNoop, WitnessEnv};
use jolt_witness::{RandomAccessRows, WitnessBundle, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::mem::MaybeUninit;
use std::sync::{Arc, Mutex};
use tracing::Span;

pub(crate) const SOURCE_HEADER: &str = include_str!("native.metal");
const SOURCE_CAP: usize = 12 << 30;
const SCRATCH_CAP: usize = 64 << 20;
const _: () = assert!(size_of::<NativeCore>() == 64 && size_of::<NativeLookup>() == 24);
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub(crate) struct NativeCore {
    pub(crate) rs1: u64,
    pub(crate) pc: u64,
    pub(crate) rs2: u64,
    pub(crate) rd: u64,
    pub(crate) imm_lo: u64,
    pub(crate) imm_hi: u64,
    pub(crate) lookup: u64,
    pub(crate) flags: u32,
    reserved: u32,
}
impl NativeCore {
    pub(crate) fn operand(&self, column: u32) -> F {
        match column {
            0 => F::from_u64(self.rs1),
            1 => F::from_u64(self.pc),
            2 => F::from_u64(self.rs2),
            3 => F::from_i128((u128::from(self.imm_lo) | (u128::from(self.imm_hi) << 64)) as i128),
            _ => F::from_u64(self.rd),
        }
    }
    pub(crate) fn product_flag(&self, index: usize) -> F {
        F::from_bool(self.flags & (1 << (index + 5)) != 0)
    }
}
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub(crate) struct NativeLookup {
    pub(crate) left: u64,
    pub(crate) right_lo: u64,
    pub(crate) right_hi: u64,
}
impl NativeLookup {
    pub(crate) fn right(&self) -> F {
        F::from_u128(u128::from(self.right_lo) | (u128::from(self.right_hi) << 64))
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
struct NativeBundle {
    core: NativeCore,
    lookup: NativeLookup,
}
impl WitnessBundle for NativeBundle {
    fn from_row(
        row: &JoltTraceRow,
        next: Option<&JoltTraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let input = InstructionInputRow::from_row(row, next, env)?;
        let (_, (left, right), output) = lookup_values(row);
        let instruction = row.instruction_flags();
        let circuit = row.circuit_flags();
        let product_flags = [
            circuit[CircuitFlags::Jump],
            circuit[CircuitFlags::WriteLookupOutputToRD],
            instruction[InstructionFlags::Branch],
            NextIsNoop::extract(row, next, env)?.0,
            circuit[CircuitFlags::VirtualInstruction],
        ];
        let flags = input
            .flags()
            .into_iter()
            .chain(product_flags)
            .enumerate()
            .fold(0, |bits, (i, value)| bits | (u32::from(value) << i));
        let imm = input.imm.0 as u128;
        Ok(Self {
            core: NativeCore {
                rs1: input.rs1_value.0,
                pc: input.unexpanded_pc.0,
                rs2: input.rs2_value.0,
                rd: input.rd_write_value.0,
                imm_lo: imm as u64,
                imm_hi: (imm >> 64) as u64,
                lookup: output,
                flags,
                reserved: 0,
            },
            lookup: NativeLookup {
                left,
                right_lo: right as u64,
                right_hi: (right >> 64) as u64,
            },
        })
    }
    fn annotated_ids() -> Vec<JoltPolynomialId> {
        Vec::new()
    }
}
pub(crate) struct CoreSource {
    pub(crate) buffer: DeviceBuffer<NativeCore>,
    pub(crate) identity: RandomAccessRows,
    pub(crate) prefix: usize,
    pub(crate) cycles: usize,
    pub(crate) tail: NativeCore,
}
pub(crate) type SharedCore = Arc<Mutex<CoreSource>>;
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct ForInput(
    #[cfg_attr(feature = "allocative", allocative(skip))] pub(crate) SharedCore,
);
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct ForClaimReduction(
    #[cfg_attr(feature = "allocative", allocative(skip))] pub(crate) Arc<Stage2Source>,
);
pub(crate) struct OpeningClaims {
    pub(crate) point: Vec<F>,
    pub(crate) values: [F; 10],
}
pub(crate) struct Stage2Source {
    pub(crate) core: SharedCore,
    pub(crate) lookup: Mutex<DeviceBuffer<NativeLookup>>,
    pub(crate) tail_lookup: NativeLookup,
    pub(crate) claims: Mutex<Option<OpeningClaims>>,
}
impl Stage2Source {
    /// Preflight reserves nothing. Capacity failure declines before publication;
    /// every actual allocation still checks the current working set.
    #[tracing::instrument(name = "metal_native_operands_capture", skip_all, fields(prefix = tracing::field::Empty, bytes = tracing::field::Empty))]
    pub(crate) fn capture(
        device: &Device,
        identity: RandomAccessRows,
        cycles: usize,
        min_prefix: usize,
    ) -> Result<Option<Arc<Self>>, KernelError<F>> {
        if identity.cycles() != cycles {
            return Ok(None);
        }
        let prefix = identity.constant_suffix_start::<NativeBundle>(cycles)?;
        let Some(bytes) = prefix.checked_add(1).and_then(|n| n.checked_mul(88)) else {
            return Ok(None);
        };
        let Some(peak) = prefix
            .checked_add(2)
            .and_then(|n| n.checked_mul(152))
            .and_then(|n| n.checked_add(SCRATCH_CAP))
        else {
            return Ok(None);
        };
        if prefix < min_prefix
            || prefix >= u32::MAX as usize
            || bytes > SOURCE_CAP
            || (prefix + 1) * 64 > device.limits().max_buffer_length
        {
            return Ok(None);
        }
        if let Err(error) = device.check_additional_peak(peak) {
            if error.class() == ErrorClass::Capacity {
                return Ok(None);
            }
            return Err(compute_error(error).into());
        }
        let _ = Span::current().record("prefix", prefix);
        let _ = Span::current().record("bytes", bytes);
        let stored = prefix + 1;
        let allocate = || -> Result<_, MetalError> {
            Ok((
                DeviceBuffer::<MaybeUninit<NativeCore>>::uninit(device, stored)?,
                DeviceBuffer::<MaybeUninit<NativeLookup>>::uninit(device, stored)?,
            ))
        };
        let (mut core, mut lookup) = match allocate() {
            Ok(buffers) => buffers,
            Err(error) if error.class() == ErrorClass::Capacity => return Ok(None),
            Err(error) => return Err(compute_error(error).into()),
        };
        // One extra row supplies odd endpoints without materializing the suffix.
        let tail = identity.window::<NativeBundle>(prefix.min(cycles - 1))?;
        let fill = |(index, (core, lookup)): (
            usize,
            (&mut MaybeUninit<NativeCore>, &mut MaybeUninit<NativeLookup>),
        )|
         -> Result<(), WitnessError> {
            let row = if index < prefix {
                identity.window::<NativeBundle>(index)?
            } else {
                tail
            };
            let _ = core.write(row.core);
            let _ = lookup.write(row.lookup);
            Ok(())
        };
        let core_out = core.write_uninit().map_err(compute_error)?;
        let lookup_out = lookup.write_uninit().map_err(compute_error)?;
        #[cfg(feature = "parallel")]
        core_out
            .par_iter_mut()
            .zip(lookup_out.par_iter_mut())
            .enumerate()
            .try_for_each(fill)?;
        #[cfg(not(feature = "parallel"))]
        for pair in core_out.iter_mut().zip(lookup_out).enumerate() {
            fill(pair)?;
        }
        // SAFETY: all indexed outputs and reserved bytes were written, including
        // the tail representative. No GPU work has borrowed these buffers.
        let (core, lookup) = unsafe {
            (
                core.assume_init().map_err(compute_error)?,
                lookup.assume_init().map_err(compute_error)?,
            )
        };
        Ok(Some(Arc::new(Self {
            core: Arc::new(Mutex::new(CoreSource {
                buffer: core,
                identity,
                prefix,
                cycles,
                tail: tail.core,
            })),
            lookup: Mutex::new(lookup),
            tail_lookup: tail.lookup,
            claims: Mutex::new(None),
        })))
    }
}
pub(crate) fn invalid(reason: &str) -> MetalError {
    MetalError::InvalidDispatch {
        pipeline: "shared native operands".into(),
        reason: reason.into(),
    }
}
pub(crate) fn compute_error(error: MetalError) -> SumcheckError<F> {
    SumcheckError::Compute {
        backend: "Metal native operands",
        reason: error.to_string(),
    }
}
