use cudarc::driver::{CudaSlice, PushKernelArg};
use jolt_field::Field;
use jolt_riscv::{CircuitFlags, InstructionFlags as InstructionFlagKind};
use jolt_witness::backend::cuda::{circuit_flag_bit, instruction_flag_bit};

use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device::DeviceFrVec;
use crate::KernelError;

pub const COLUMNS: usize = 5;

pub const UNEXPANDED_PC: usize = 0;
pub const PC: usize = 1;
pub const IS_VIRTUAL: usize = 2;
pub const IS_FIRST_IN_SEQUENCE: usize = 3;
pub const IS_NOOP: usize = 4;

pub fn upload_from_device<F: Field>(
    context: &CudaKernelContext,
    address: &CudaSlice<u64>,
    pc_words: &CudaSlice<u32>,
    flags: &CudaSlice<u32>,
    cycles: usize,
) -> Result<Vec<DeviceFrVec>, KernelError<F>> {
    if cycles == 0 || address.len() < cycles || pc_words.len() < cycles || flags.len() < cycles {
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

    let mut unexpanded_pc = context.alloc(cycles)?;
    let mut pc = context.alloc(cycles)?;
    let mut is_virtual = context.alloc(cycles)?;
    let mut is_first_in_sequence = context.alloc(cycles)?;
    let mut is_noop = context.alloc(cycles)?;
    let mut unmapped = context.upload_u64_slice(&[u64::MAX])?;

    let count = CudaKernelContext::count_of(cycles)?;
    let mut builder = context.stream().launch_builder(context.ss_columns_device());
    let _ = builder.arg(address);
    let _ = builder.arg(pc_words);
    let _ = builder.arg(flags);
    let _ = builder.arg(unexpanded_pc.limbs_mut());
    let _ = builder.arg(pc.limbs_mut());
    let _ = builder.arg(is_virtual.limbs_mut());
    let _ = builder.arg(is_first_in_sequence.limbs_mut());
    let _ = builder.arg(is_noop.limbs_mut());
    let _ = builder.arg(&virtual_bit);
    let _ = builder.arg(&first_bit);
    let _ = builder.arg(&noop_bit);
    let _ = builder.arg(&mut unmapped);
    let _ = builder.arg(&count);
    // SAFETY: thread `t < cycles` reads `address[t]`, `pc_words[t]` and
    // `flags[t]`, all checked above to hold at least `cycles` entries, and
    // writes `out[t*LIMBS..t*LIMBS+4]` of each of the five output buffers, every
    // one a fresh `cycles`-element allocation distinct from the inputs and from
    // each other. `unmapped` is a single element mutated only through
    // `atomicMin`. Threads with `t >= cycles` return before any access.
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

    Ok(vec![
        unexpanded_pc,
        pc,
        is_virtual,
        is_first_in_sequence,
        is_noop,
    ])
}
