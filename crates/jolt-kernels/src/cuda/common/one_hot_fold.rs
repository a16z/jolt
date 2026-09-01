use std::sync::Arc;

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use jolt_field::Field;

use jolt_witness::JoltWitnessPlane;

use super::context::{context_for, CudaKernelContext, BLOCK};
use super::device::{require_fr_slice, DeviceFrVec};
use super::device_columns::{device_trace_columns, DeviceTraceColumns};
use super::devices::{fan_out, witness_windows, CycleWindow, DeviceTask};
use super::error::CudaError;
use crate::{KernelError, ProofSession};

pub const LANES: usize = 8;

pub const PACKED_BITS: usize = 32;

pub const SHARED_BUDGET: usize = 32 * 1024;

pub const CYCLES_PER_THREAD: usize = 32;

pub const POLYS_PER_BLOCK: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoldTuning {
    pub polys_per_block: usize,
    pub cycles_per_thread: usize,
    pub shared_budget: usize,
}

impl Default for FoldTuning {
    fn default() -> Self {
        Self {
            polys_per_block: POLYS_PER_BLOCK,
            cycles_per_thread: CYCLES_PER_THREAD,
            shared_budget: SHARED_BUDGET,
        }
    }
}

pub struct DeviceOneHotColumns {
    lookup: Arc<CudaSlice<u64>>,
    pc: Arc<CudaSlice<u32>>,
    ram: Arc<CudaSlice<u32>>,
    families: [usize; 3],
    chunk_bits: usize,
    cycles: usize,
}

pub(crate) struct OneHotShards {
    windows: Vec<CycleWindow>,
    columns: Vec<DeviceOneHotColumns>,
}

enum FoldPart {
    Resident(DeviceFrVec),
    Limbs(Vec<u64>),
}

impl OneHotShards {
    pub(crate) fn new<F: Field>(
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        cycles: usize,
        families: [usize; 3],
        chunk_bits: usize,
        addresses: usize,
    ) -> Result<Self, KernelError<F>> {
        let windows = witness_windows(cycles);
        let mut columns = Vec::with_capacity(windows.len());
        for (ordinal, window) in windows.iter().enumerate() {
            let device = context_for(ordinal).ok_or(KernelError::InvariantViolation {
                reason: "a one-hot cycle-fold window names an absent device",
            })?;
            let raw = device_trace_columns::<F>(
                device, session, witness, cycles, window, families, addresses,
            )?;
            columns.push(DeviceOneHotColumns::from_device(
                raw, families, chunk_bits, window.len,
            )?);
        }
        Ok(Self { windows, columns })
    }

    pub(crate) fn from_windows(
        windows: Vec<CycleWindow>,
        columns: Vec<DeviceOneHotColumns>,
    ) -> Result<Self, CudaError> {
        if windows.len() != columns.len() || windows.is_empty() {
            return Err(CudaError::LengthMismatch {
                expected: windows.len(),
                got: columns.len(),
            });
        }
        Ok(Self { windows, columns })
    }

    #[cfg(test)]
    pub(crate) fn single(columns: DeviceOneHotColumns) -> Self {
        let window = CycleWindow {
            start: 0,
            len: columns.cycles,
        };
        Self {
            windows: vec![window],
            columns: vec![columns],
        }
    }

    pub(crate) fn whole(&self) -> Result<&DeviceOneHotColumns, CudaError> {
        self.columns.first().ok_or(CudaError::InvariantViolation {
            reason: "a one-hot shard set holds no columns",
        })
    }

    pub(crate) fn fold<F: Field>(
        &self,
        cycle_point: &[F],
        tuning: FoldTuning,
    ) -> Result<DeviceFrVec, CudaError> {
        let shards = self.windows.len();
        let absent = || CudaError::InvariantViolation {
            reason: "a one-hot cycle-fold window names an absent device",
        };
        if shards <= 1 {
            let context = context_for(0).ok_or_else(absent)?;
            return self.whole()?.fold_cycles(context, cycle_point, tuning);
        }
        let point = require_fr_slice(cycle_point)?;
        let tasks: Vec<DeviceTask<'_, FoldPart, CudaError>> = (0..shards)
            .map(|ordinal| {
                let task: DeviceTask<'_, FoldPart, CudaError> = Box::new(move || {
                    let context = context_for(ordinal).ok_or_else(absent)?;
                    let columns =
                        self.columns
                            .get(ordinal)
                            .ok_or(CudaError::InvariantViolation {
                                reason: "a one-hot cycle-fold window has no columns",
                            })?;
                    let eq = tracing::info_span!(
                        "cuda_one_hot_fold_eq",
                        device = ordinal,
                        cycles = columns.cycles,
                    )
                    .in_scope(|| context.eq_evals_shard(point, ordinal, shards))?;
                    let folded = tracing::info_span!(
                        "cuda_one_hot_fold_window",
                        device = ordinal,
                        addresses = columns.addresses(),
                        polys = columns.polys(),
                        cycles = columns.cycles,
                        shared =
                            columns.addresses() * LANES * size_of::<u64>() <= tuning.shared_budget,
                    )
                    .in_scope(|| columns.fold_cycles_with_eq(context, &eq, tuning))?;
                    if ordinal == 0 {
                        return Ok(FoldPart::Resident(folded));
                    }
                    tracing::info_span!(
                        "cuda_one_hot_fold_download",
                        device = ordinal,
                        elements = folded.len(),
                    )
                    .in_scope(|| Ok(FoldPart::Limbs(context.download_u64(folded.limbs())?)))
                });
                task
            })
            .collect();
        let parts = fan_out(tasks)?;
        tracing::info_span!("cuda_one_hot_fold_reduce", shards).in_scope(
            || -> Result<DeviceFrVec, CudaError> {
                let context = context_for(0).ok_or_else(absent)?;
                let mut parts = parts.into_iter();
                let mut total = match parts.next() {
                    Some(FoldPart::Resident(resident)) => resident,
                    _ => {
                        return Err(CudaError::InvariantViolation {
                            reason: "the one-hot cycle fold lost its device-0 window",
                        })
                    }
                };
                for part in parts {
                    let FoldPart::Limbs(limbs) = part else {
                        return Err(CudaError::InvariantViolation {
                            reason: "a one-hot cycle-fold window past device 0 stayed resident",
                        });
                    };
                    let addend = context.upload_limbs(&limbs)?;
                    if addend.len() != total.len() {
                        return Err(CudaError::LengthMismatch {
                            expected: total.len(),
                            got: addend.len(),
                        });
                    }
                    total = context.add(&total, &addend)?;
                }
                Ok(total)
            },
        )
    }
}

impl DeviceOneHotColumns {
    pub fn new(
        context: &CudaKernelContext,
        lookup: &[u64],
        pc: &[u32],
        ram: &[u32],
        families: [usize; 3],
        chunk_bits: usize,
        cycles: usize,
    ) -> Result<Self, CudaError> {
        let lookup = if lookup.is_empty() {
            context.alloc_u64(0)?
        } else {
            context.upload_u64_slice(lookup)?
        };
        let pc = if pc.is_empty() {
            context.alloc_u32(0)?
        } else {
            context.upload_u32_slice(pc)?
        };
        let ram = if ram.is_empty() {
            context.alloc_u32(0)?
        } else {
            context.upload_u32_slice(ram)?
        };
        Self::from_device(
            DeviceTraceColumns {
                lookup: Arc::new(lookup),
                pc: Arc::new(pc),
                ram: Arc::new(ram),
            },
            families,
            chunk_bits,
            cycles,
        )
    }

    pub(crate) fn from_device(
        columns: DeviceTraceColumns,
        families: [usize; 3],
        chunk_bits: usize,
        cycles: usize,
    ) -> Result<Self, CudaError> {
        let DeviceTraceColumns { lookup, pc, ram } = columns;
        let polys = families.iter().sum::<usize>();
        if polys == 0 || chunk_bits == 0 {
            return Err(CudaError::InvariantViolation {
                reason: "a one-hot cycle fold needs at least one polynomial and one chunk bit",
            });
        }
        if !cycles.is_power_of_two() {
            return Err(CudaError::InvariantViolation {
                reason: "a one-hot cycle fold needs a power-of-two cycle count",
            });
        }
        if chunk_bits * families[1] > PACKED_BITS || chunk_bits * families[2] > PACKED_BITS {
            return Err(CudaError::NotImplemented {
                kernel: "the CUDA one-hot cycle fold packs the bytecode PC and the remapped RAM \
                         word address into one 32-bit word each",
            });
        }
        if families[0] > 0 && lookup.len() < 2 * cycles {
            return Err(CudaError::LengthMismatch {
                expected: 2 * cycles,
                got: lookup.len(),
            });
        }
        if families[1] > 0 && pc.len() < cycles {
            return Err(CudaError::LengthMismatch {
                expected: cycles,
                got: pc.len(),
            });
        }
        if families[2] > 0 && ram.len() < cycles {
            return Err(CudaError::LengthMismatch {
                expected: cycles,
                got: ram.len(),
            });
        }
        Ok(Self {
            lookup,
            pc,
            ram,
            families,
            chunk_bits,
            cycles,
        })
    }

    pub const fn polys(&self) -> usize {
        self.families[0] + self.families[1] + self.families[2]
    }

    pub const fn addresses(&self) -> usize {
        1usize << self.chunk_bits
    }

    pub fn fold_cycles<F: Field>(
        &self,
        context: &CudaKernelContext,
        cycle_point: &[F],
        tuning: FoldTuning,
    ) -> Result<DeviceFrVec, CudaError> {
        if cycle_point.len() != self.cycles.ilog2() as usize {
            return Err(CudaError::LengthMismatch {
                expected: self.cycles.ilog2() as usize,
                got: cycle_point.len(),
            });
        }
        let eq = context.eq_evals(require_fr_slice(cycle_point)?)?;
        self.fold_cycles_with_eq(context, &eq, tuning)
    }

    pub fn fold_cycles_with_eq(
        &self,
        context: &CudaKernelContext,
        eq: &DeviceFrVec,
        tuning: FoldTuning,
    ) -> Result<DeviceFrVec, CudaError> {
        if eq.len() != self.cycles {
            return Err(CudaError::LengthMismatch {
                expected: self.cycles,
                got: eq.len(),
            });
        }
        if tuning.polys_per_block == 0 || tuning.cycles_per_thread == 0 {
            return Err(CudaError::InvariantViolation {
                reason: "a one-hot cycle fold needs a positive block width and tile size",
            });
        }

        let polys = self.polys();
        let addresses = self.addresses();
        let bucket_bytes = addresses * LANES * size_of::<u64>();
        let (per_block, shared_bytes) = if bucket_bytes <= tuning.shared_budget {
            let fit = (tuning.shared_budget / bucket_bytes).max(1);
            let per_block = tuning.polys_per_block.min(fit).min(polys);
            (per_block, per_block * bucket_bytes)
        } else {
            (tuning.polys_per_block.min(polys), 0)
        };

        let mut slots = context.alloc_u64(polys * addresses * LANES)?;
        let groups = CudaKernelContext::count_of(polys.div_ceil(per_block))?;
        let threads = self.cycles.div_ceil(tuning.cycles_per_thread);
        let blocks = CudaKernelContext::count_of(threads)?.div_ceil(BLOCK).max(1);

        let instruction = CudaKernelContext::count_of(self.families[0])?;
        let bytecode = CudaKernelContext::count_of(self.families[1])?;
        let ram = CudaKernelContext::count_of(self.families[2])?;
        let bits = CudaKernelContext::count_of(self.chunk_bits)?;
        let address_count = CudaKernelContext::count_of(addresses)?;
        let cycle_count = CudaKernelContext::count_of(self.cycles)?;
        let width = CudaKernelContext::count_of(per_block)?;
        let use_shared = u32::from(shared_bytes > 0);

        let mut builder = context.stream().launch_builder(context.ohf_fold());
        let _ = builder.arg(self.lookup.as_ref());
        let _ = builder.arg(self.pc.as_ref());
        let _ = builder.arg(self.ram.as_ref());
        let _ = builder.arg(&instruction);
        let _ = builder.arg(&bytecode);
        let _ = builder.arg(&ram);
        let _ = builder.arg(&bits);
        let _ = builder.arg(&address_count);
        let _ = builder.arg(&cycle_count);
        let _ = builder.arg(eq.limbs());
        let _ = builder.arg(&mut slots);
        let _ = builder.arg(&width);
        let _ = builder.arg(&use_shared);
        // SAFETY: block `(x, y)` owns the polynomials `[y * width, y * width + count)`
        // of `polys` and strides over the cycle column, so every thread reads
        // `eq[j]` for `j < cycles` (inside `eq`'s `cycles` elements, checked above),
        // its family's source column at cycle `j` — `lookup[2j]`/`lookup[2j + 1]` of
        // `2 * cycles`, or `pc[j]`/`ram[j]` of `cycles`, whichever the family selects
        // (each present column's length is checked in `new`, and a family with no
        // polynomials is never selected) — and accumulates into
        // `slots[(p * addresses + a) * LANES + lane]` with `a` masked below
        // `addresses` and `p < polys`, inside `slots`'s `polys * addresses * LANES`
        // u64s. Concurrent accumulation into one bucket is `atomicAdd` on
        // 32-bit-halved lanes, so no carry is lost. When `use_shared` the same
        // accumulation lands in `count * addresses * LANES` u64s of dynamic shared
        // memory, matching `shared_mem_bytes`, and is flushed to the same global
        // slots after a `__syncthreads()` every thread reaches (the only early
        // return is uniform across the block).
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, groups, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: CudaKernelContext::count_of(shared_bytes)?,
            })
        }?;
        context.stream().synchronize()?;

        let buckets = CudaKernelContext::count_of(polys * addresses)?;
        let mut folded = context.alloc(polys * addresses)?;
        let mut builder = context.stream().launch_builder(context.ohf_reduce());
        let _ = builder.arg(&slots);
        let _ = builder.arg(folded.limbs_mut());
        let _ = builder.arg(&buckets);
        // SAFETY: thread `b < buckets` reads its own `LANES` u64s of `slots`, whose
        // length is `buckets * LANES` by construction, and writes only `out[b]` of
        // `buckets` field elements; `out` is a fresh allocation distinct from `slots`.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(buckets)) }?;
        context.stream().synchronize()?;
        Ok(folded)
    }
}

pub fn affine_table(
    context: &CudaKernelContext,
    base: u64,
    stride: u64,
    len: usize,
) -> Result<DeviceFrVec, CudaError> {
    let mut table = context.alloc(len)?;
    let count = CudaKernelContext::count_of(len)?;
    let mut builder = context.stream().launch_builder(context.ohf_affine());
    let _ = builder.arg(&base);
    let _ = builder.arg(&stride);
    let _ = builder.arg(table.limbs_mut());
    let _ = builder.arg(&count);
    // SAFETY: thread `i < count` reads the two by-value scalars and writes only
    // `out[i]` of `count` field elements, a fresh allocation. Threads with
    // `i >= count` return before any access.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    context.stream().synchronize()?;
    Ok(table)
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::EqPolynomial;
    use proptest::prelude::*;

    use super::super::context::shared_context;
    use super::super::testing::fr;
    use super::{affine_table, DeviceOneHotColumns, FoldTuning, POLYS_PER_BLOCK, SHARED_BUDGET};
    use crate::cuda::common::pack::COLD;

    fn mix(seed: u64, cycle: usize, salt: u64) -> u64 {
        let value = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(cycle as u64 + salt);
        value.wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ (value >> 29)
    }

    struct Columns {
        lookup: Vec<u64>,
        pc: Vec<u32>,
        ram: Vec<u32>,
    }

    fn columns(seed: u64, cycles: usize, span: u32, cold_every: usize) -> Columns {
        let mut lookup = Vec::with_capacity(2 * cycles);
        let mut pc = Vec::with_capacity(cycles);
        let mut ram = Vec::with_capacity(cycles);
        for cycle in 0..cycles {
            lookup.push(mix(seed, cycle, 1));
            lookup.push(mix(seed, cycle, 2));
            pc.push(if cycle % cold_every == 1 {
                COLD
            } else {
                (mix(seed, cycle, 3) % u64::from(span)) as u32
            });
            ram.push(if cycle.is_multiple_of(cold_every) {
                COLD
            } else {
                (mix(seed, cycle, 4) % u64::from(span)) as u32
            });
        }
        Columns { lookup, pc, ram }
    }

    fn expected_fold(
        columns: &Columns,
        cycles: usize,
        chunk_bits: usize,
        families: [usize; 3],
        cycle_point: &[Fr],
    ) -> Vec<Fr> {
        let eq = EqPolynomial::new(cycle_point.to_vec()).evaluations();
        let addresses = 1usize << chunk_bits;
        let mut folded = vec![Fr::from_u64(0); families.iter().sum::<usize>() * addresses];
        let mut poly = 0;
        for (family, count) in families.into_iter().enumerate() {
            for local in 0..count {
                let shift = chunk_bits * (count - 1 - local);
                for (cycle, weight) in eq.iter().enumerate().take(cycles) {
                    let index = match family {
                        0 => {
                            let wide = u128::from(columns.lookup[2 * cycle])
                                | (u128::from(columns.lookup[2 * cycle + 1]) << 64);
                            Some(((wide >> shift) as usize) & (addresses - 1))
                        }
                        1 => (columns.pc[cycle] != COLD)
                            .then(|| ((columns.pc[cycle] >> shift) as usize) & (addresses - 1)),
                        _ => (columns.ram[cycle] != COLD)
                            .then(|| ((columns.ram[cycle] >> shift) as usize) & (addresses - 1)),
                    };
                    if let Some(index) = index {
                        folded[poly * addresses + index] += *weight;
                    }
                }
                poly += 1;
            }
        }
        folded
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(6))]
        #[test]
        fn one_hot_cycle_fold_matches_cpu(
            log_t in 4usize..9,
            chunk_bits in prop::sample::select(vec![2usize, 4, 6]),
            bytecode_polys in 1usize..3,
            ram_polys in 1usize..3,
            cold_every in 2usize..5,
            seed in any::<u64>(),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            if chunk_bits * bytecode_polys > 32 || chunk_bits * ram_polys > 32 {
                return Ok(());
            }
            let cycles = 1usize << log_t;
            let families = [64 / chunk_bits, bytecode_polys, ram_polys];
            let span = 1u32 << (chunk_bits * bytecode_polys.max(ram_polys) - 1).min(31);
            let packed = columns(seed, cycles, span, cold_every);
            let cycle_point: Vec<Fr> = (0..log_t).map(|i| fr(seed ^ (i as u64 * 17 + 3))).collect();

            let expected = expected_fold(&packed, cycles, chunk_bits, families, &cycle_point);
            let uploaded = DeviceOneHotColumns::new(
                context,
                &packed.lookup,
                &packed.pc,
                &packed.ram,
                families,
                chunk_bits,
                cycles,
            )
            .expect("upload one-hot columns");
            for (polys_per_block, shared_budget) in
                [(1usize, SHARED_BUDGET), (POLYS_PER_BLOCK, SHARED_BUDGET), (32, SHARED_BUDGET), (1, 0)]
            {
                let got = uploaded
                    .fold_cycles(
                        context,
                        &cycle_point,
                        FoldTuning { polys_per_block, shared_budget, ..FoldTuning::default() },
                    )
                    .expect("device one-hot cycle fold")
                    .to_host()
                    .expect("download the folded tables");
                prop_assert_eq!(
                    got,
                    expected.clone(),
                    "the fold diverged at polys_per_block {} shared_budget {}",
                    polys_per_block,
                    shared_budget
                );
            }
        }
    }

    #[test]
    fn windowed_folds_sum_to_the_whole_cycle_domain() {
        let Some(context) = shared_context() else {
            return;
        };
        const LOG_T: usize = 9;
        let cycles = 1usize << LOG_T;
        let chunk_bits = 4;
        let families = [64 / chunk_bits, 2, 2];
        let packed = columns(11, cycles, 1u32 << (chunk_bits * 2 - 1), 3);
        let cycle_point: Vec<Fr> = (0..LOG_T).map(|i| fr(31 + 7 * i as u64)).collect();

        let build = |base: usize, len: usize| {
            DeviceOneHotColumns::new(
                context,
                &packed.lookup[base * 2..(base + len) * 2],
                &packed.pc[base..base + len],
                &packed.ram[base..base + len],
                families,
                chunk_bits,
                len,
            )
            .expect("upload one-hot columns")
        };

        let whole = build(0, cycles)
            .fold_cycles(context, &cycle_point, FoldTuning::default())
            .expect("whole-domain fold")
            .to_host()
            .expect("download");

        for shards in [2usize, 4, 8] {
            let len = cycles / shards;
            let mut summed = vec![Fr::from_u64(0); whole.len()];
            for shard in 0..shards {
                let eq = context
                    .eq_evals_shard(&cycle_point, shard, shards)
                    .expect("eq shard");
                let part = build(shard * len, len)
                    .fold_cycles_with_eq(context, &eq, FoldTuning::default())
                    .expect("windowed fold")
                    .to_host()
                    .expect("download");
                for (total, addend) in summed.iter_mut().zip(&part) {
                    *total += *addend;
                }
            }
            assert_eq!(
                summed, whole,
                "the one-hot cycle fold over {shards} windows must sum to the whole-domain fold: \
                 every address bucket is a sum over the cycles that land in it, and eq_evals_shard \
                 supplies each window its slice of the cycle eq table",
            );
        }
    }

    #[test]
    fn hot_index_fold_matches_cpu_over_cold_cycles() {
        let Some(context) = shared_context() else {
            return;
        };
        let log_t = 6;
        let cycles = 1usize << log_t;
        let cycle_point: Vec<Fr> = (0..log_t).map(|i| fr(i as u64 * 31 + 11)).collect();
        let eq = EqPolynomial::new(cycle_point.clone()).evaluations();

        for chunk_bits in [4usize, 12] {
            let addresses = 1usize << chunk_bits;
            let hot: Vec<Option<usize>> = (0..cycles)
                .map(|cycle| {
                    if cycle.is_multiple_of(5) {
                        None
                    } else {
                        Some((cycle * 7) % addresses)
                    }
                })
                .collect();
            let mut expected = vec![Fr::from_u64(0); addresses];
            for (address, weight) in hot.iter().zip(&eq) {
                if let Some(address) = address {
                    expected[*address] += *weight;
                }
            }

            let words: Vec<u32> = hot
                .iter()
                .map(|address| address.map_or(COLD, |address| address as u32))
                .collect();
            let got = DeviceOneHotColumns::new(
                context,
                &[],
                &[],
                &words,
                [0, 0, 1],
                chunk_bits,
                words.len(),
            )
            .expect("upload packed addresses")
            .fold_cycles(context, &cycle_point, FoldTuning::default())
            .expect("device one-hot cycle fold")
            .to_host()
            .expect("download the folded table");
            assert_eq!(
                got, expected,
                "the fold diverged at chunk_bits {chunk_bits}"
            );
        }
    }

    #[test]
    fn affine_table_matches_cpu() {
        let Some(context) = shared_context() else {
            return;
        };
        let len = 1usize << 7;
        let expected: Vec<Fr> = (0..len as u64)
            .map(|k| Fr::from_u64(8 * k + 1234))
            .collect();
        let got = affine_table(context, 1234, 8, len)
            .expect("device affine table")
            .to_host()
            .expect("download the affine table");
        assert_eq!(got, expected);
    }
}
