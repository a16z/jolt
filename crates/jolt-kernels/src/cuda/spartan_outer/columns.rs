use cudarc::driver::{CudaSlice, PushKernelArg};
use jolt_field::Field;

use super::witness::{
    self, Packed, FIRST_IN_SEQUENCE_BIT, NARROW, NEXT_IS_FIRST_IN_SEQUENCE_BIT,
    NEXT_IS_VIRTUAL_BIT, NEXT_PC_SLOT, NEXT_UNEXPANDED_PC_SLOT, PC_SLOT, UNEXPANDED_PC_SLOT,
    VIRTUAL_INSTRUCTION_BIT, WIDE,
};
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::error::CudaError;

pub struct DeviceR1csInputs {
    narrow: CudaSlice<u64>,
    wide: CudaSlice<u64>,
    flags: CudaSlice<u32>,
    layout: CudaSlice<u32>,
    cycles: usize,
}

impl DeviceR1csInputs {
    pub fn new(context: &CudaKernelContext, packed: &Packed) -> Result<Self, CudaError> {
        let cycles = packed.flags.len();
        if cycles == 0
            || packed.narrow.len() != cycles * NARROW
            || packed.wide.len() != cycles * WIDE * 2
        {
            return Err(CudaError::InvariantViolation {
                reason: "the packed R1CS inputs must hold one entry per cycle per column",
            });
        }

        let mut narrow = context.upload_u64_slice(&packed.narrow)?;
        let wide = context.upload_u64_slice(&packed.wide)?;
        let raw = context.upload_u32_slice(&packed.flags)?;
        let layout = context.upload_u32_slice(&witness::LAYOUT)?;
        let mut flags = context.alloc_u32(cycles)?;

        let count = CudaKernelContext::count_of(cycles)?;
        let pc = CudaKernelContext::count_of(PC_SLOT)?;
        let unexpanded_pc = CudaKernelContext::count_of(UNEXPANDED_PC_SLOT)?;
        let next_pc = CudaKernelContext::count_of(NEXT_PC_SLOT)?;
        let next_unexpanded_pc = CudaKernelContext::count_of(NEXT_UNEXPANDED_PC_SLOT)?;
        let mut builder = context.stream().launch_builder(context.so_shift());
        let _ = builder.arg(&raw);
        let _ = builder.arg(&mut narrow);
        let _ = builder.arg(&mut flags);
        let _ = builder.arg(&pc);
        let _ = builder.arg(&unexpanded_pc);
        let _ = builder.arg(&next_pc);
        let _ = builder.arg(&next_unexpanded_pc);
        let _ = builder.arg(&VIRTUAL_INSTRUCTION_BIT);
        let _ = builder.arg(&FIRST_IN_SEQUENCE_BIT);
        let _ = builder.arg(&NEXT_IS_VIRTUAL_BIT);
        let _ = builder.arg(&NEXT_IS_FIRST_IN_SEQUENCE_BIT);
        let _ = builder.arg(&count);
        // SAFETY: thread `t < cycles` reads `raw[t]` and, when `t + 1 < cycles`,
        // `raw[t + 1]` — both inside `raw`'s `cycles` entries — plus the two source
        // slots of row `t + 1`, inside `narrow`'s `cycles * NARROW` u64s. It writes
        // `flags[t]` (a fresh allocation, one slot per thread) and slots
        // `NEXT_PC_SLOT` / `NEXT_UNEXPANDED_PC_SLOT` of row `t`. Those two write
        // slots are disjoint from the two read slots (`PC_SLOT`,
        // `UNEXPANDED_PC_SLOT`), which no thread writes, so the in-place shift
        // cannot read a value another thread has already overwritten.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;

        Ok(Self {
            narrow,
            wide,
            flags,
            layout,
            cycles,
        })
    }

    pub const fn cycles(&self) -> usize {
        self.cycles
    }

    pub const fn narrow(&self) -> &CudaSlice<u64> {
        &self.narrow
    }

    pub const fn wide(&self) -> &CudaSlice<u64> {
        &self.wide
    }

    pub const fn flags(&self) -> &CudaSlice<u32> {
        &self.flags
    }

    pub const fn layout(&self) -> &CudaSlice<u32> {
        &self.layout
    }
}

pub struct LinearForms {
    offsets: Vec<u32>,
    counts: Vec<u32>,
    terms: Vec<u32>,
    coefficients: Vec<jolt_field::Fr>,
    constants: Vec<jolt_field::Fr>,
}

impl LinearForms {
    pub fn new() -> Self {
        Self {
            offsets: Vec::new(),
            counts: Vec::new(),
            terms: Vec::new(),
            coefficients: Vec::new(),
            constants: Vec::new(),
        }
    }

    pub fn push<F: Field>(&mut self, weights: &[F], constant: F) -> Result<(), CudaError> {
        if weights.len() != witness::VARIABLES {
            return Err(CudaError::LengthMismatch {
                expected: witness::VARIABLES,
                got: weights.len(),
            });
        }
        let start = u32::try_from(self.terms.len()).map_err(|_| CudaError::InvariantViolation {
            reason: "the Spartan outer linear forms exceed a u32 term offset",
        })?;
        let mut count = 0u32;
        for (variable, &weight) in weights.iter().enumerate() {
            if weight.is_zero() {
                continue;
            }
            self.terms.push(witness::LAYOUT[variable]);
            self.coefficients
                .push(crate::cuda::common::device::require_fr(weight)?);
            count += 1;
        }
        self.offsets.push(start);
        self.counts.push(count);
        self.constants
            .push(crate::cuda::common::device::require_fr(constant)?);
        Ok(())
    }

    pub fn upload(&self, context: &CudaKernelContext) -> Result<DeviceLinearForms, CudaError> {
        Ok(DeviceLinearForms {
            offsets: context.upload_u32_slice(&self.offsets)?,
            counts: context.upload_u32_slice(&self.counts)?,
            terms: context.upload_u32_slice(&self.terms)?,
            coefficients: context.upload(&self.coefficients)?,
            constants: context.upload(&self.constants)?,
        })
    }
}

pub struct DeviceLinearForms {
    pub offsets: CudaSlice<u32>,
    pub counts: CudaSlice<u32>,
    pub terms: CudaSlice<u32>,
    pub coefficients: crate::cuda::common::device::DeviceFrVec,
    pub constants: crate::cuda::common::device::DeviceFrVec,
}

impl DeviceLinearForms {
    pub fn bind_args<'a>(&'a self, builder: &mut cudarc::driver::LaunchArgs<'a>) {
        let _ = builder.arg(&self.offsets);
        let _ = builder.arg(&self.counts);
        let _ = builder.arg(self.constants.limbs());
        let _ = builder.arg(&self.terms);
        let _ = builder.arg(self.coefficients.limbs());
    }
}
