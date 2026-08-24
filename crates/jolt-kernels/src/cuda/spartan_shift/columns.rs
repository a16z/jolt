use std::sync::Arc;

use cudarc::driver::{CudaSlice, PushKernelArg};
use jolt_field::Field;
use jolt_riscv::{CircuitFlags, InstructionFlags as InstructionFlagKind};
use jolt_witness::backend::cuda::{circuit_flag_bit, instruction_flag_bit, DeviceTrace};

use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::half_fold::{NarrowColumn, NarrowKind};
use crate::cuda::common::prefix_suffix::NarrowColumns;
use crate::KernelError;

pub const COLUMNS: usize = 5;

pub const UNEXPANDED_PC: usize = 0;
pub const PC: usize = 1;
pub const IS_VIRTUAL: usize = 2;
pub const IS_FIRST_IN_SEQUENCE: usize = 3;
pub const IS_NOOP: usize = 4;

const FLAG_BASE: u32 = 32;

pub struct ShiftColumns {
    trace: Arc<DeviceTrace>,
    packed: CudaSlice<u64>,
    entries: usize,
}

impl NarrowColumns for ShiftColumns {
    fn count(&self) -> usize {
        COLUMNS
    }

    fn entries(&self) -> usize {
        self.entries
    }

    fn column(&self, index: usize) -> Option<NarrowColumn<'_>> {
        let (words, kind) = match index {
            UNEXPANDED_PC => (self.trace.unexpanded_pc(), NarrowKind::U64),
            PC => (&self.packed, NarrowKind::U32),
            IS_VIRTUAL => (&self.packed, NarrowKind::Bit(FLAG_BASE)),
            IS_FIRST_IN_SEQUENCE => (&self.packed, NarrowKind::Bit(FLAG_BASE + 1)),
            IS_NOOP => (&self.packed, NarrowKind::Bit(FLAG_BASE + 2)),
            _ => return None,
        };
        Some(NarrowColumn::packed(words, kind, self.entries))
    }

    #[cfg(feature = "allocative")]
    fn device_bytes(&self) -> usize {
        self.packed.len() * size_of::<u64>()
    }
}

pub fn pack_from_device<F: Field>(
    context: &CudaKernelContext,
    trace: Arc<DeviceTrace>,
    pc_words: &CudaSlice<u32>,
    flags: &CudaSlice<u32>,
    cycles: usize,
) -> Result<ShiftColumns, KernelError<F>> {
    if cycles == 0
        || trace.unexpanded_pc().len() < cycles
        || pc_words.len() < cycles
        || flags.len() < cycles
    {
        return Err(KernelError::InvalidGeometry {
            reason: format!("the device shift sources do not cover {cycles} cycles"),
        });
    }
    let virtual_bit = circuit_flag_bit(CircuitFlags::VirtualInstruction).ok_or(
        KernelError::InvariantViolation {
            reason: "VirtualInstruction has no canonical device flag bit",
        },
    )?;
    let first_bit = circuit_flag_bit(CircuitFlags::IsFirstInSequence).ok_or(
        KernelError::InvariantViolation {
            reason: "IsFirstInSequence has no canonical device flag bit",
        },
    )?;
    let noop_bit = instruction_flag_bit(InstructionFlagKind::IsNoop).ok_or(
        KernelError::InvariantViolation {
            reason: "IsNoop has no canonical device flag bit",
        },
    )?;

    let mut packed = context.alloc_u64(cycles)?;
    let mut unmapped = context.upload_u64_slice(&[u64::MAX])?;

    let count = CudaKernelContext::count_of(cycles)?;
    let mut builder = context.stream().launch_builder(context.ss_packed_columns());
    let _ = builder.arg(pc_words);
    let _ = builder.arg(flags);
    let _ = builder.arg(&mut packed);
    let _ = builder.arg(&virtual_bit);
    let _ = builder.arg(&first_bit);
    let _ = builder.arg(&noop_bit);
    let _ = builder.arg(&FLAG_BASE);
    let _ = builder.arg(&mut unmapped);
    let _ = builder.arg(&count);
    // SAFETY: thread `t < cycles` reads `pc_words[t]` and `flags[t]`, both
    // checked above to hold at least `cycles` entries, and writes `packed[t]` of
    // a fresh `cycles`-word allocation distinct from either input. `unmapped` is
    // a single element mutated only through `atomicMin`. The three source bits
    // come from `circuit_flag_bit`/`instruction_flag_bit`, all below 32, and the
    // destination bits `FLAG_BASE ..= FLAG_BASE + 2` are below 64, so every
    // shift stays in range. Threads with `t >= cycles` return before any access.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }
        .map_err(crate::cuda::common::error::CudaError::from)?;
    context
        .stream()
        .synchronize()
        .map_err(crate::cuda::common::error::CudaError::from)?;

    let unmapped = context.download_u64(&unmapped)?;
    if unmapped.first().is_some_and(|&cycle| cycle != u64::MAX) {
        return Err(KernelError::InvariantViolation {
            reason: "a cycle has no bytecode PC mapping, so the shift Pc column is undefined",
        });
    }

    Ok(ShiftColumns {
        trace,
        packed,
        entries: cycles,
    })
}
