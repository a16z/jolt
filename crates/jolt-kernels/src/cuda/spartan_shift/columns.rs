use crate::cuda::common::error::CudaError;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use jolt_field::JoltField;
use jolt_riscv::{CircuitFlags, InstructionFlags as InstructionFlagKind};
use jolt_witness::backend::cuda::{circuit_flag_bit, instruction_flag_bit, DeviceTrace};

use crate::cuda::common::context::{CudaKernelContext, BLOCK};
use crate::cuda::common::device::{DeviceFrVec, LIMBS};
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

    fn fused_suffix_folds(
        &self,
        context: &CudaKernelContext,
        weights: &DeviceFrVec,
    ) -> Result<Option<Vec<DeviceFrVec>>, CudaError> {
        fold_suffix_columns(
            context,
            self.trace.unexpanded_pc(),
            &self.packed,
            self.entries,
            weights,
        )
        .map(Some)
    }

    #[cfg(feature = "allocative")]
    fn device_bytes(&self) -> usize {
        self.packed.len() * size_of::<u64>()
    }
}

pub fn pack_from_device<F: JoltField>(
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
        .map_err(CudaError::from)?;
    context.stream().synchronize().map_err(CudaError::from)?;

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

fn fold_suffix_columns(
    context: &CudaKernelContext,
    unexpanded_pc: &CudaSlice<u64>,
    packed: &CudaSlice<u64>,
    entries: usize,
    weights: &DeviceFrVec,
) -> Result<Vec<DeviceFrVec>, CudaError> {
    if entries == 0
        || weights.is_empty()
        || !entries.is_multiple_of(weights.len())
        || unexpanded_pc.len() < entries
        || packed.len() < entries
    {
        return Err(CudaError::InvariantViolation {
            reason:
                "shift suffix folds require complete rows and a non-empty dividing weight table",
        });
    }
    let out_len = entries / weights.len();
    let sum_count = CudaKernelContext::count_of(weights.len())?;
    let blocks = CudaKernelContext::count_of(out_len)?;
    let mut out = (0..COLUMNS)
        .map(|_| context.alloc(out_len))
        .collect::<Result<Vec<_>, _>>()?;
    let mut builder = context.stream().launch_builder(context.ss_fold_suffix());
    let _ = builder.arg(unexpanded_pc);
    let _ = builder.arg(packed);
    let _ = builder.arg(weights.limbs());
    for column in &mut out {
        let _ = builder.arg(column.limbs_mut());
    }
    let _ = builder.arg(&sum_count);
    let _ = builder.arg(&FLAG_BASE);
    // SAFETY: block a reads rows a * sum_count + b for b < sum_count;
    // both native columns cover entries = out_len * sum_count. Weights cover
    // sum_count fields. All five outputs have out_len fields and are distinct.
    // The fixed flag bits are below 64. Every thread reaches every barrier;
    // the power-of-two block reduction uses BLOCK * LIMBS shared u64s.
    let _ = unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
        })
    }?;
    Ok(out)
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use super::{fold_suffix_columns, COLUMNS, FLAG_BASE};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::fr;
    use jolt_field::{Fr, Ring, Zero};

    #[test]
    fn suffix_folds_match_native_dot_products() {
        let Some(context) = shared_context() else {
            return;
        };
        // Uneven blocks, full-width PCs, every flag combination, and the
        // per-thread accumulator batch boundary are independent failure cases.
        for sum_len in [1, 17, 4096, 4096 * 256 + 1] {
            let out_len = if sum_len > 4096 { 1 } else { 3 };
            let entries = sum_len * out_len;
            let pc: Vec<u64> = (0..entries)
                .map(|i| (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
                .collect();
            let packed: Vec<u64> = (0..entries)
                .map(|i| (!pc[i] & u64::from(u32::MAX)) | (((i % 8) as u64) << FLAG_BASE))
                .collect();
            let weights: Vec<Fr> = (0..sum_len).map(|i| fr(i as u64 + 71)).collect();
            let d_pc = context.upload_u64_slice(&pc).expect("upload pc");
            let d_packed = context.upload_u64_slice(&packed).expect("upload packed");
            let d_weights = context.upload(&weights).expect("upload weights");
            let result = fold_suffix_columns(context, &d_pc, &d_packed, entries, &d_weights)
                .expect("fold suffix");
            let mut expected = vec![[Fr::zero(); COLUMNS]; out_len];
            for (a, totals) in expected.iter_mut().enumerate() {
                for (b, weight) in weights.iter().enumerate() {
                    let i = a * sum_len + b;
                    let values = [
                        pc[i],
                        packed[i] & u64::from(u32::MAX),
                        (packed[i] >> FLAG_BASE) & 1,
                        (packed[i] >> (FLAG_BASE + 1)) & 1,
                        (packed[i] >> (FLAG_BASE + 2)) & 1,
                    ];
                    for (total, value) in totals.iter_mut().zip(values) {
                        *total += *weight * Fr::from_u64(value);
                    }
                }
            }
            for (column, got) in result.iter().enumerate() {
                let expected: Vec<Fr> = expected.iter().map(|row| row[column]).collect();
                assert_eq!(
                    got.to_host().expect("download"),
                    expected,
                    "sum_len={sum_len}"
                );
            }
        }
    }
}
