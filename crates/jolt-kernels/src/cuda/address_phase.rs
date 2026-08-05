#![expect(
    dead_code,
    reason = "implementation target: the device address phase wires this once its kernels land"
)]

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use jolt_field::{Fr, MulPow2};
use jolt_lookup_tables::tables::LookupTableKind;
use jolt_lookup_tables::XLEN as RISCV_XLEN;

use super::context::{CudaKernelContext, BLOCK};
use super::device::{DeviceFrVec, LIMBS};
use super::error::CudaError;

pub const CHUNK_LEN: usize = 8;
pub const CHUNK_SIZE: usize = 1 << CHUNK_LEN;

pub const NO_TABLE: u32 = u32::MAX;
const SKIP: u32 = u32::MAX;
const RAF_LANES: usize = 6;
const MAX_SUFFIXES: usize = 4;

pub struct DeviceRows {
    lookup_index: CudaSlice<u64>,
    table_index: CudaSlice<u32>,
    raf_flag: CudaSlice<u8>,
    cycles: usize,
}

impl DeviceRows {
    pub fn new(
        context: &CudaKernelContext,
        lookup_index: &[u128],
        table_index: &[Option<usize>],
        raf_flag: &[bool],
    ) -> Result<Self, CudaError> {
        let cycles = lookup_index.len();
        if table_index.len() != cycles || raf_flag.len() != cycles {
            return Err(CudaError::LengthMismatch {
                expected: cycles,
                got: table_index.len().min(raf_flag.len()),
            });
        }
        let mut bits = Vec::with_capacity(cycles * 2);
        for &index in lookup_index {
            bits.push(index as u64);
            bits.push((index >> 64) as u64);
        }
        let tables: Vec<u32> = table_index
            .iter()
            .map(|slot| slot.map_or(NO_TABLE, |index| index as u32))
            .collect();
        let flags: Vec<u8> = raf_flag.iter().map(|&flag| u8::from(flag)).collect();
        Ok(Self {
            lookup_index: context.upload_u64_slice(&bits)?,
            table_index: context.upload_u32_slice(&tables)?,
            raf_flag: context.upload_u8_slice(&flags)?,
            cycles,
        })
    }

    pub const fn cycles(&self) -> usize {
        self.cycles
    }
}

pub struct RafBuckets {
    pub shift_half: DeviceFrVec,
    pub shift_full: DeviceFrVec,
    pub left: DeviceFrVec,
    pub right: DeviceFrVec,
    pub identity: DeviceFrVec,
    pub upper_all_ones: DeviceFrVec,
}

struct Segments {
    order: CudaSlice<u32>,
    offsets: CudaSlice<u32>,
    counts: CudaSlice<u32>,
}

fn segment_rows(
    context: &CudaKernelContext,
    keys: &CudaSlice<u32>,
    rows: usize,
    buckets: usize,
) -> Result<Segments, CudaError> {
    let count = CudaKernelContext::count_of(rows)?;
    let mut counts = context.alloc_u32(buckets)?;
    let mut builder = context.stream().launch_builder(context.ap_histogram());
    let _ = builder.arg(keys);
    let _ = builder.arg(&mut counts);
    let _ = builder.arg(&count);
    // SAFETY: thread `j < rows` reads only `keys[j]` of the `rows`-element key
    // buffer and, unless the key is `SKIP`, atomically increments
    // `counts[keys[j]]`. Every non-`SKIP` key is `< buckets` by construction in
    // the two key kernels, and `counts` holds `buckets` u32s zeroed by
    // `alloc_u32`. Concurrent increments are `atomicAdd`, so the accumulation is
    // race-free; threads with `j >= rows` return first.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    context.stream().synchronize()?;

    let histogram = context.download_u32(&counts)?;
    let total: usize = histogram.iter().map(|&value| value as usize).sum();
    let offsets = context.exclusive_scan_u32(&histogram)?;
    let offsets = context.upload_u32_slice(&offsets)?;

    let mut cursors = context.alloc_u32(buckets)?;
    let mut order = context.alloc_u32(total.max(1))?;
    let mut builder = context.stream().launch_builder(context.ap_scatter());
    let _ = builder.arg(keys);
    let _ = builder.arg(&offsets);
    let _ = builder.arg(&mut cursors);
    let _ = builder.arg(&mut order);
    let _ = builder.arg(&count);
    // SAFETY: thread `j < rows` reads `keys[j]`, and for a non-`SKIP` key claims
    // a unique slot via `atomicAdd(&cursors[key], 1)` before writing
    // `order[offsets[key] + slot]`. `offsets` is the exclusive scan of the
    // histogram just computed from the same keys, so per key the claimed slots
    // run over `[0, counts[key])` and `offsets[key] + slot` stays inside that
    // key's segment; segments partition `[0, total)` and `order` holds
    // `max(total, 1)` u32s. Distinct keys own disjoint segments and the atomic
    // makes slots unique within a key, so every write goes to a distinct index.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    context.stream().synchronize()?;

    Ok(Segments {
        order,
        offsets,
        counts,
    })
}

pub fn init_raf_buckets(
    context: &CudaKernelContext,
    rows: &DeviceRows,
    u_evals: &DeviceFrVec,
    address_bits: usize,
    phase: usize,
) -> Result<RafBuckets, CudaError> {
    if u_evals.len() != rows.cycles {
        return Err(CudaError::LengthMismatch {
            expected: rows.cycles,
            got: u_evals.len(),
        });
    }
    let suffix_len = suffix_len(address_bits, phase)?;
    let count = CudaKernelContext::count_of(rows.cycles)?;
    let suffix_len_arg = CudaKernelContext::count_of(suffix_len)?;

    let mut keys = context.alloc_u32(rows.cycles)?;
    let mut builder = context.stream().launch_builder(context.ap_raf_keys());
    let _ = builder.arg(&rows.lookup_index);
    let _ = builder.arg(&suffix_len_arg);
    let _ = builder.arg(&mut keys);
    let _ = builder.arg(&count);
    // SAFETY: thread `j < rows.cycles` reads `lookup_index[2j]`/`[2j+1]` of a
    // `2 * cycles` buffer and writes only `keys[j]` of `cycles`. The written
    // value is masked to `CHUNK_SIZE - 1`, so it is a valid bucket index.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    context.stream().synchronize()?;

    let segments = segment_rows(context, &keys, rows.cycles, CHUNK_SIZE)?;

    let upper_suffix_bits = suffix_len.saturating_sub(address_bits / 2);
    let canonical = u32::from(
        jolt_claims::protocols::jolt::geometry::instruction::CANONICAL_INSTRUCTION_ADDRESS,
    );
    let upper_suffix_bits = CudaKernelContext::count_of(upper_suffix_bits)?;

    let mut buckets = context.alloc(RAF_LANES * CHUNK_SIZE)?;
    let chunk_count = CudaKernelContext::count_of(CHUNK_SIZE)?;
    let mut builder = context.stream().launch_builder(context.ap_raf_reduce());
    let _ = builder.arg(&segments.order);
    let _ = builder.arg(&segments.offsets);
    let _ = builder.arg(&segments.counts);
    let _ = builder.arg(&rows.lookup_index);
    let _ = builder.arg(&rows.raf_flag);
    let _ = builder.arg(u_evals.limbs());
    let _ = builder.arg(&suffix_len_arg);
    let _ = builder.arg(&upper_suffix_bits);
    let _ = builder.arg(&canonical);
    let _ = builder.arg(buckets.limbs_mut());
    // SAFETY: block `bucket < CHUNK_SIZE` reads its own segment
    // `order[offsets[bucket] .. + counts[bucket]]` — the partition built by
    // `segment_rows` from keys `< CHUNK_SIZE` — and each element is a row index
    // `< cycles`, so the `lookup_index` (`2 * cycles`), `raf_flag` (`cycles`) and
    // `u_evals` (`cycles`) reads are in range. Writes: thread 0 writes
    // `buckets[lane * CHUNK_SIZE + bucket]` for `lane < RAF_LANES`, one slot per
    // (lane, bucket) of `RAF_LANES * CHUNK_SIZE`. Shared memory is
    // `BLOCK * LIMBS` u64s matching `shared_mem_bytes`, with `__syncthreads()`
    // between tree levels and after each lane's write.
    let _ = unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (chunk_count, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
        })
    }?;
    context.stream().synchronize()?;

    let one = Fr::from(1u64);
    let half_scale = context.upload(&[one.mul_pow_2(suffix_len / 2)])?;
    let full_scale = context.upload(&[one.mul_pow_2(suffix_len)])?;
    let mut builder = context.stream().launch_builder(context.ap_scale_shift());
    let _ = builder.arg(buckets.limbs_mut());
    let _ = builder.arg(half_scale.limbs());
    let _ = builder.arg(full_scale.limbs());
    // SAFETY: thread `i < CHUNK_SIZE` read-modify-writes exactly `buckets[i]` and
    // `buckets[CHUNK_SIZE + i]` — the two shift lanes, one thread per element so
    // uncontended — and reads the two single-element scale buffers. `buckets`
    // holds `RAF_LANES * CHUNK_SIZE` elements.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(chunk_count)) }?;
    context.stream().synchronize()?;

    let lane = |index: usize| buckets.slice_elements(index * CHUNK_SIZE, CHUNK_SIZE);
    Ok(RafBuckets {
        shift_half: lane(0)?,
        shift_full: lane(1)?,
        left: lane(2)?,
        right: lane(3)?,
        identity: lane(4)?,
        upper_all_ones: lane(5)?,
    })
}

pub fn init_suffix_buckets(
    context: &CudaKernelContext,
    rows: &DeviceRows,
    u_evals: &DeviceFrVec,
    present: &[LookupTableKind<RISCV_XLEN>],
    address_bits: usize,
    phase: usize,
) -> Result<Vec<Vec<DeviceFrVec>>, CudaError> {
    if u_evals.len() != rows.cycles {
        return Err(CudaError::LengthMismatch {
            expected: rows.cycles,
            got: u_evals.len(),
        });
    }
    if present.is_empty() {
        return Ok(Vec::new());
    }
    let suffix_len = suffix_len(address_bits, phase)?;

    let table_count = LookupTableKind::<RISCV_XLEN>::COUNT;
    let mut table_slots = vec![SKIP; table_count];
    let mut suffix_ids = Vec::new();
    let mut suffix_offsets = Vec::with_capacity(present.len());
    let mut suffix_counts = Vec::with_capacity(present.len());
    for (slot, table) in present.iter().enumerate() {
        let index = table.index();
        if index >= table_count {
            return Err(CudaError::LengthMismatch {
                expected: table_count,
                got: index,
            });
        }
        table_slots[index] = slot as u32;
        let suffixes = table.suffixes();
        if suffixes.len() > MAX_SUFFIXES {
            return Err(CudaError::InvariantViolation {
                reason: "a lookup table declares more suffixes than the device kernel supports",
            });
        }
        suffix_offsets.push(suffix_ids.len() as u32);
        suffix_counts.push(suffixes.len() as u32);
        suffix_ids.extend(suffixes.iter().map(|suffix| *suffix as u32));
    }

    let count = CudaKernelContext::count_of(rows.cycles)?;
    let suffix_len_arg = CudaKernelContext::count_of(suffix_len)?;
    let table_count_arg = CudaKernelContext::count_of(table_count)?;
    let device_slots = context.upload_u32_slice(&table_slots)?;

    let mut keys = context.alloc_u32(rows.cycles)?;
    let mut builder = context.stream().launch_builder(context.ap_table_keys());
    let _ = builder.arg(&rows.lookup_index);
    let _ = builder.arg(&rows.table_index);
    let _ = builder.arg(&device_slots);
    let _ = builder.arg(&table_count_arg);
    let _ = builder.arg(&suffix_len_arg);
    let _ = builder.arg(&mut keys);
    let _ = builder.arg(&count);
    // SAFETY: thread `j < rows.cycles` reads `lookup_index[2j]`/`[2j+1]` of a
    // `2 * cycles` buffer, `table_index[j]` of `cycles`, and — only after
    // bounds-checking it against `table_count` — `table_slots[table]` of
    // `table_count`. It writes only `keys[j]` of `cycles`, either `SKIP` or
    // `slot * CHUNK_SIZE + chunk` with `slot < present.len()` and
    // `chunk < CHUNK_SIZE`, so a live key is `< present.len() * CHUNK_SIZE`.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    context.stream().synchronize()?;

    let bucket_count = present.len() * CHUNK_SIZE;
    let segments = segment_rows(context, &keys, rows.cycles, bucket_count)?;

    let device_suffix_ids = context.upload_u32_slice(&suffix_ids)?;
    let device_suffix_offsets = context.upload_u32_slice(&suffix_offsets)?;
    let device_suffix_counts = context.upload_u32_slice(&suffix_counts)?;

    let mut buckets = context.alloc(suffix_ids.len() * CHUNK_SIZE)?;
    let blocks = CudaKernelContext::count_of(bucket_count)?;
    let mut builder = context.stream().launch_builder(context.ap_suffix_reduce());
    let _ = builder.arg(&segments.order);
    let _ = builder.arg(&segments.offsets);
    let _ = builder.arg(&segments.counts);
    let _ = builder.arg(&rows.lookup_index);
    let _ = builder.arg(u_evals.limbs());
    let _ = builder.arg(&device_suffix_ids);
    let _ = builder.arg(&device_suffix_offsets);
    let _ = builder.arg(&device_suffix_counts);
    let _ = builder.arg(&suffix_len_arg);
    let _ = builder.arg(buckets.limbs_mut());
    // SAFETY: block `blockIdx.x < present.len() * CHUNK_SIZE` derives
    // `slot = blockIdx.x / CHUNK_SIZE < present.len()`, so its reads of
    // `suffix_offsets`/`suffix_counts` (both `present.len()` u32s) are in range,
    // and `suffix_ids[family_base + s]` for `s < suffix_counts[slot]` stays
    // inside the flattened `suffix_ids` by construction. It reads its own
    // segment of `order` (the partition from `segment_rows` over the same
    // bucket count), whose elements are row indices `< cycles`, bounding the
    // `lookup_index` and `u_evals` reads. Writes: thread 0 writes
    // `buckets[(family_base + s) * CHUNK_SIZE + bucket]`, one slot per
    // (family, bucket) of `suffix_ids.len() * CHUNK_SIZE`, and distinct blocks
    // have distinct `(slot, bucket)` hence distinct targets. `acc` is indexed by
    // `s < families <= MAX_SUFFIXES`, its declared extent. Shared memory is
    // `BLOCK * LIMBS` u64s matching `shared_mem_bytes`, with `__syncthreads()`
    // between tree levels and after each family's write.
    let _ = unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
        })
    }?;
    context.stream().synchronize()?;

    let mut tables = Vec::with_capacity(present.len());
    for (slot, _) in present.iter().enumerate() {
        let base = suffix_offsets[slot] as usize;
        let families = suffix_counts[slot] as usize;
        let mut columns = Vec::with_capacity(families);
        for family in 0..families {
            columns.push(buckets.slice_elements((base + family) * CHUNK_SIZE, CHUNK_SIZE)?);
        }
        tables.push(columns);
    }
    Ok(tables)
}

pub fn condense_u_evals(
    context: &CudaKernelContext,
    rows: &DeviceRows,
    u_evals: &mut DeviceFrVec,
    v_prev: &DeviceFrVec,
    address_bits: usize,
    phase: usize,
) -> Result<(), CudaError> {
    if u_evals.len() != rows.cycles {
        return Err(CudaError::LengthMismatch {
            expected: rows.cycles,
            got: u_evals.len(),
        });
    }
    if v_prev.len() != CHUNK_SIZE {
        return Err(CudaError::LengthMismatch {
            expected: CHUNK_SIZE,
            got: v_prev.len(),
        });
    }
    if phase == 0 {
        return Err(CudaError::InvariantViolation {
            reason: "the first address phase has no previous phase to condense",
        });
    }
    let suffix_len = suffix_len(address_bits, phase - 1)?;
    let count = CudaKernelContext::count_of(rows.cycles)?;
    let suffix_len_arg = CudaKernelContext::count_of(suffix_len)?;

    let mut builder = context.stream().launch_builder(context.ap_condense());
    let _ = builder.arg(&rows.lookup_index);
    let _ = builder.arg(u_evals.limbs_mut());
    let _ = builder.arg(v_prev.limbs());
    let _ = builder.arg(&suffix_len_arg);
    let _ = builder.arg(&count);
    // SAFETY: thread `j < rows.cycles` reads `lookup_index[2j]`/`[2j+1]` of a
    // `2 * cycles` buffer and `v_prev[chunk]` with `chunk` masked to
    // `CHUNK_SIZE - 1` (and `v_prev` length-checked as `CHUNK_SIZE` above), then
    // read-modify-writes exactly `u_evals[j]` of `cycles`. The update is
    // in-place but one thread per element, so it is uncontended, and no thread
    // reads another's target.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    context.stream().synchronize()?;
    Ok(())
}

fn suffix_len(address_bits: usize, phase: usize) -> Result<usize, CudaError> {
    address_bits
        .checked_sub((phase + 1) * CHUNK_LEN)
        .ok_or(CudaError::InvariantViolation {
            reason: "address phase index exceeds the address width",
        })
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::instruction::{
        InstructionReadRafDimensions, CANONICAL_INSTRUCTION_ADDRESS,
    };
    use jolt_field::Fr;
    use jolt_lookup_tables::tables::LookupTableKind;
    use jolt_lookup_tables::XLEN as RISCV_XLEN;
    use jolt_poly::Polynomial;
    use jolt_witness::witnesses::{InstructionRafFlag, LookupIndex, TableIndex};
    use proptest::prelude::*;
    use std::num::NonZeroUsize;

    use super::super::context::{shared_context, CudaKernelContext};
    use super::super::testing::fr;
    use super::{
        condense_u_evals, init_raf_buckets, init_suffix_buckets, DeviceRows, CHUNK_LEN, CHUNK_SIZE,
    };
    use crate::reference::instruction_read_raf::{
        InstructionReadRafKernel, InstructionReadRafWitness,
    };

    const ADDRESS_BITS: usize = 128;
    const PHASES: usize = ADDRESS_BITS / CHUNK_LEN;

    fn rows(log_t: usize, seed: u64) -> Vec<InstructionReadRafWitness> {
        let tables: Vec<LookupTableKind<RISCV_XLEN>> =
            <LookupTableKind<RISCV_XLEN> as strum::IntoEnumIterator>::iter().collect();
        (0..1usize << log_t)
            .map(|j| {
                let mixed = (j as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(seed);
                let index = (u128::from(mixed) << 61) | u128::from(mixed.rotate_left(17));
                InstructionReadRafWitness {
                    lookup_index: LookupIndex(index),
                    table_index: TableIndex(if mixed.is_multiple_of(11) {
                        None
                    } else {
                        Some(tables[(mixed % tables.len() as u64) as usize].index())
                    }),
                    raf_flag: InstructionRafFlag(mixed.is_multiple_of(3)),
                }
            })
            .collect()
    }

    fn reference_at_phase(log_t: usize, seed: u64, phase: usize) -> InstructionReadRafKernel<Fr> {
        let dimensions =
            InstructionReadRafDimensions::new(log_t, ADDRESS_BITS, NonZeroUsize::new(8).unwrap());
        let r_reduction: Vec<Fr> = (0..log_t).map(|i| fr(seed + i as u64 + 3)).collect();
        let mut kernel = InstructionReadRafKernel::new(
            dimensions,
            &r_reduction,
            rows(log_t, seed),
            fr(seed + 1),
        )
        .expect("reference kernel");
        for round in 0..phase * CHUNK_LEN {
            kernel
                .bind(fr(seed + round as u64 + 71))
                .expect("reference bind");
        }
        kernel
    }

    fn device_rows(context: &CudaKernelContext, rows: &[InstructionReadRafWitness]) -> DeviceRows {
        let indices: Vec<u128> = rows.iter().map(|row| row.lookup_index.0).collect();
        let tables: Vec<Option<usize>> = rows.iter().map(|row| row.table_index.0).collect();
        let flags: Vec<bool> = rows.iter().map(|row| row.raf_flag.0).collect();
        DeviceRows::new(context, &indices, &tables, &flags).expect("device rows")
    }

    fn evals(poly: &Polynomial<Fr>) -> Vec<Fr> {
        poly.evals().to_vec()
    }

    proptest! {
        #[test]
        fn raf_buckets_match_the_reference_init_phase(
            log_t in 4usize..=10,
            seed in any::<u64>(),
            phase in 0usize..PHASES,
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let host = reference_at_phase(log_t, seed, phase);
            let rows = device_rows(context, host.rows());
            let u_evals = context.upload(host.u_evals()).expect("upload u_evals");

            let got = init_raf_buckets(context, &rows, &u_evals, ADDRESS_BITS, phase)
                .expect("device init_raf_buckets");

            for (label, device, expected) in [
                ("shift_half", &got.shift_half, evals(&host.raf_left.q_shift)),
                ("left", &got.left, evals(&host.raf_left.q_value)),
                ("shift_full", &got.shift_full, evals(&host.raf_identity.q_shift)),
                ("identity", &got.identity, evals(&host.raf_identity.q_value)),
                ("right", &got.right, evals(&host.raf_right.q_value)),
            ] {
                prop_assert_eq!(
                    &device.to_host().expect("download"),
                    &expected,
                    "{} bucket diverged at phase {}",
                    label,
                    phase
                );
            }

            if CANONICAL_INSTRUCTION_ADDRESS {
                prop_assert_eq!(
                    &got.upper_all_ones.to_host().expect("download"),
                    &evals(&host.raf_upper_all_ones.q_shift),
                    "upper_all_ones bucket diverged at phase {}",
                    phase
                );
            } else {
                prop_assert_eq!(
                    host.raf_upper_all_ones.q_shift.evals().len(),
                    1,
                    "the reference built an upper_all_ones table with akita off, so the \
                     device lane is no longer untested and this test must compare it",
                );
            }
        }

        #[test]
        fn suffix_buckets_match_the_reference_init_phase(
            log_t in 4usize..=10,
            seed in any::<u64>(),
            phase in 0usize..PHASES,
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let host = reference_at_phase(log_t, seed, phase);
            let rows = device_rows(context, host.rows());
            let u_evals = context.upload(host.u_evals()).expect("upload u_evals");
            let present: Vec<LookupTableKind<RISCV_XLEN>> =
                host.suffix_tables.iter().map(|(table, _)| *table).collect();

            let got = init_suffix_buckets(
                context, &rows, &u_evals, &present, ADDRESS_BITS, phase,
            )
            .expect("device init_suffix_buckets");

            prop_assert_eq!(got.len(), host.suffix_tables.len());
            for (columns, (table, expected)) in got.iter().zip(&host.suffix_tables) {
                prop_assert_eq!(columns.len(), expected.len());
                for (slot, (column, want)) in columns.iter().zip(expected).enumerate() {
                    prop_assert_eq!(
                        &column.to_host().expect("download"),
                        &evals(want),
                        "table {:?} suffix slot {} diverged at phase {}",
                        table,
                        slot,
                        phase
                    );
                }
            }
        }

        #[test]
        fn condense_u_evals_matches_the_reference_condensation(
            log_t in 4usize..=10,
            seed in any::<u64>(),
            phase in 1usize..PHASES,
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let before = reference_at_phase(log_t, seed, phase - 1);
            let after = reference_at_phase(log_t, seed, phase);

            let rows = device_rows(context, before.rows());
            let mut u_evals = context.upload(before.u_evals()).expect("upload u_evals");
            let v_prev = context
                .upload(after.v_table(phase - 1))
                .expect("upload v_prev");

            condense_u_evals(context, &rows, &mut u_evals, &v_prev, ADDRESS_BITS, phase)
                .expect("device condense_u_evals");

            prop_assert_eq!(
                &u_evals.to_host().expect("download"),
                &after.u_evals().to_vec(),
                "condensed u_evals diverged entering phase {}",
                phase
            );
        }
    }

    #[test]
    fn raf_buckets_are_chunk_sized() {
        let Some(context) = shared_context() else {
            return;
        };
        let host = reference_at_phase(8, 5, 0);
        let rows = device_rows(context, host.rows());
        let u_evals = context.upload(host.u_evals()).expect("upload u_evals");
        let got = init_raf_buckets(context, &rows, &u_evals, ADDRESS_BITS, 0)
            .expect("device init_raf_buckets");
        for column in [&got.shift_half, &got.shift_full, &got.left, &got.right] {
            assert_eq!(column.len(), CHUNK_SIZE);
        }
    }
}
