use std::sync::Arc;

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use jolt_field::{Fr, MulPow2};
use jolt_lookup_tables::tables::LookupTableKind;
use jolt_lookup_tables::XLEN as RISCV_XLEN;

use crate::cuda::common::context::{CudaKernelContext, BLOCK};
use crate::cuda::common::device::{DeviceFrVec, LIMBS};
use crate::cuda::common::error::CudaError;
use crate::cuda::common::unreduced::{alloc_slots, finalize_slots, fold_slot_chunks, ACCUM_LIMBS};

pub const CHUNK_LEN: usize = 8;
pub const CHUNK_SIZE: usize = 1 << CHUNK_LEN;

pub const NO_TABLE: u32 = u32::MAX;
const SKIP: u32 = u32::MAX;
const RAF_LANES: usize = 6;

pub(super) const RAF_CHUNKS: usize = 64;

pub(super) const SUFFIX_CHUNKS: usize = 64;
const MAX_SUFFIXES: usize = 4;

pub struct DeviceRows {
    lookup_index: Arc<CudaSlice<u64>>,
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
        Self::from_encoded(context, &bits, &tables, &flags)
    }

    pub fn from_encoded(
        context: &CudaKernelContext,
        bits: &[u64],
        tables: &[u32],
        flags: &[u8],
    ) -> Result<Self, CudaError> {
        let cycles = tables.len();
        if bits.len() != cycles * 2 || flags.len() != cycles {
            return Err(CudaError::LengthMismatch {
                expected: cycles,
                got: (bits.len() / 2).min(flags.len()),
            });
        }
        Self::from_device(
            context,
            Arc::new(context.upload_u64_slice(bits)?),
            tables,
            flags,
        )
    }

    pub fn from_device_columns(
        lookup_index: Arc<CudaSlice<u64>>,
        table_index: CudaSlice<u32>,
        raf_flag: CudaSlice<u8>,
        cycles: usize,
    ) -> Result<Self, CudaError> {
        if lookup_index.len() < cycles * 2 || table_index.len() < cycles || raf_flag.len() < cycles
        {
            return Err(CudaError::LengthMismatch {
                expected: cycles,
                got: table_index.len().min(raf_flag.len()),
            });
        }
        Ok(Self {
            lookup_index,
            table_index,
            raf_flag,
            cycles,
        })
    }

    pub fn from_device(
        context: &CudaKernelContext,
        lookup_index: Arc<CudaSlice<u64>>,
        tables: &[u32],
        flags: &[u8],
    ) -> Result<Self, CudaError> {
        let cycles = tables.len();
        if lookup_index.len() != cycles * 2 || flags.len() != cycles {
            return Err(CudaError::LengthMismatch {
                expected: cycles,
                got: (lookup_index.len() / 2).min(flags.len()),
            });
        }
        Ok(Self {
            lookup_index,
            table_index: context.upload_u32_slice(tables)?,
            raf_flag: context.upload_u8_slice(flags)?,
            cycles,
        })
    }

    pub const fn cycles(&self) -> usize {
        self.cycles
    }

    pub(super) fn lookup_index(&self) -> &CudaSlice<u64> {
        &self.lookup_index
    }

    pub(super) const fn table_index(&self) -> &CudaSlice<u32> {
        &self.table_index
    }

    pub(super) const fn raf_flag(&self) -> &CudaSlice<u8> {
        &self.raf_flag
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

pub(super) struct Segments {
    pub(super) order: CudaSlice<u32>,
    pub(super) offsets: CudaSlice<u32>,
    pub(super) counts: CudaSlice<u32>,
}

#[tracing::instrument(skip_all, name = "ap_segment_rows")]
pub(super) fn segment_rows(
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
    // SAFETY: thread `j < rows` reads `keys[j]` of `rows` and increments
    // `counts[keys[j]]`; live keys are `< buckets` and `counts` holds `buckets`
    // u32s. Concurrent increments are `atomicAdd`.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;

    let offsets = context.exclusive_scan_u32_on_device(&counts, buckets)?;

    let mut cursors = context.alloc_u32(buckets)?;
    let mut order = context.alloc_u32(rows.max(1))?;
    let mut builder = context.stream().launch_builder(context.ap_scatter());
    let _ = builder.arg(keys);
    let _ = builder.arg(&offsets);
    let _ = builder.arg(&mut cursors);
    let _ = builder.arg(&mut order);
    let _ = builder.arg(&count);
    // SAFETY: thread `j < rows` reads `keys[j]` of `rows` and writes
    // `order[offsets[key] + slot]`. Since `offsets` is the exclusive scan of
    // this key set's histogram and `slot` comes from `atomicAdd(&cursors[key])`,
    // slots stay in `[0, counts[key])`, so each write lands in that key's
    // segment of the partition of `[0, total)` and no two threads share a slot.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;

    Ok(Segments {
        order,
        offsets,
        counts,
    })
}

#[tracing::instrument(skip_all, name = "ap_init_raf_buckets")]
pub fn init_raf_buckets(
    context: &CudaKernelContext,
    rows: &DeviceRows,
    u_evals: &DeviceFrVec,
    address_bits: usize,
    phase: usize,
) -> Result<RafBuckets, CudaError> {
    init_raf_buckets_chunked(context, rows, u_evals, address_bits, phase, RAF_CHUNKS)
}

pub fn init_raf_buckets_chunked(
    context: &CudaKernelContext,
    rows: &DeviceRows,
    u_evals: &DeviceFrVec,
    address_bits: usize,
    phase: usize,
    chunks: usize,
) -> Result<RafBuckets, CudaError> {
    if u_evals.len() != rows.cycles {
        return Err(CudaError::LengthMismatch {
            expected: rows.cycles,
            got: u_evals.len(),
        });
    }
    if chunks == 0 {
        return Err(CudaError::InvariantViolation {
            reason: "a chunked raf reduce needs at least one chunk",
        });
    }
    let suffix_len = suffix_len(address_bits, phase)?;
    let count = CudaKernelContext::count_of(rows.cycles)?;
    let suffix_len_arg = CudaKernelContext::count_of(suffix_len)?;

    let mut keys = context.alloc_u32(rows.cycles)?;
    let mut builder = context.stream().launch_builder(context.ap_raf_keys());
    let _ = builder.arg(rows.lookup_index());
    let _ = builder.arg(&suffix_len_arg);
    let _ = builder.arg(&mut keys);
    let _ = builder.arg(&count);
    // SAFETY: thread `j < cycles` reads `lookup_index[2j]`/`[2j+1]` of
    // `2 * cycles` and writes only `keys[j]` of `cycles`.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    context.stream().synchronize()?;

    let segments = segment_rows(context, &keys, rows.cycles, CHUNK_SIZE)?;

    let upper_suffix_bits = suffix_len.saturating_sub(address_bits / 2);
    let canonical = u32::from(
        jolt_claims::protocols::jolt::geometry::instruction::CANONICAL_INSTRUCTION_ADDRESS,
    );
    let upper_suffix_bits = CudaKernelContext::count_of(upper_suffix_bits)?;

    let mut partials = alloc_slots(context, RAF_LANES * CHUNK_SIZE * chunks)?;
    let chunks_arg = CudaKernelContext::count_of(chunks)?;
    let blocks = CudaKernelContext::count_of(CHUNK_SIZE * chunks)?;
    let mut builder = context
        .stream()
        .launch_builder(context.ap_raf_reduce_chunked());
    let _ = builder.arg(&segments.order);
    let _ = builder.arg(&segments.offsets);
    let _ = builder.arg(&segments.counts);
    let _ = builder.arg(rows.lookup_index());
    let _ = builder.arg(&rows.raf_flag);
    let _ = builder.arg(u_evals.limbs());
    let _ = builder.arg(&suffix_len_arg);
    let _ = builder.arg(&upper_suffix_bits);
    let _ = builder.arg(&canonical);
    let _ = builder.arg(&chunks_arg);
    let _ = builder.arg(&mut partials);
    // SAFETY: block `b < CHUNK_SIZE * chunks` takes bucket `b / chunks` and chunk
    // `b % chunks`, so the bucket index stays below `CHUNK_SIZE` and indexes
    // `offsets`/`counts`; it reads the stride-`chunks * blockDim.x` subsequence of
    // `order[offsets[bucket] .. + counts[bucket]]`, whose elements are row indices
    // `< cycles`, bounding the `lookup_index` (`2 * cycles`), `raf_flag` and
    // `u_evals` (`cycles`) reads. Thread 0 writes the `2 * ACCUM_LIMBS` lanes at
    // `partials[((lane * CHUNK_SIZE + bucket) * chunks + chunk) * 2 * ACCUM_LIMBS]`
    // for `lane < RAF_LANES` — one disjoint slot per (lane, bucket, chunk) of the
    // `RAF_LANES * CHUNK_SIZE * chunks` reserved by `alloc_slots`. Shared memory is
    // `BLOCK * 2 * ACCUM_LIMBS` u64s, the folded accumulator width the block
    // reduction operates on, matching `shared_mem_bytes`.
    let _ = unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: BLOCK * 2 * ACCUM_LIMBS as u32 * size_of::<u64>() as u32,
        })
    }?;
    context.stream().synchronize()?;

    let slots = fold_slot_chunks(context, &partials, RAF_LANES * CHUNK_SIZE, chunks)?;
    let mut buckets = finalize_slots(context, &slots, RAF_LANES * CHUNK_SIZE)?;

    let one = Fr::from(1u64);
    let half_scale = context.upload(&[one.mul_pow_2(suffix_len / 2)])?;
    let full_scale = context.upload(&[one.mul_pow_2(suffix_len)])?;
    let mut builder = context.stream().launch_builder(context.ap_scale_shift());
    let _ = builder.arg(buckets.limbs_mut());
    let _ = builder.arg(half_scale.limbs());
    let _ = builder.arg(full_scale.limbs());
    let chunk_count = CudaKernelContext::count_of(CHUNK_SIZE)?;
    // SAFETY: thread `i < CHUNK_SIZE` read-modify-writes exactly `buckets[i]` and
    // `buckets[CHUNK_SIZE + i]` of `RAF_LANES * CHUNK_SIZE` — one thread per
    // element, so uncontended — and reads the two single-element scale buffers.
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

#[tracing::instrument(skip_all, name = "ap_init_suffix_buckets")]
pub fn init_suffix_buckets(
    context: &CudaKernelContext,
    rows: &DeviceRows,
    u_evals: &DeviceFrVec,
    present: &[LookupTableKind<RISCV_XLEN>],
    address_bits: usize,
    phase: usize,
) -> Result<Vec<Vec<DeviceFrVec>>, CudaError> {
    init_suffix_buckets_chunked(
        context,
        rows,
        u_evals,
        present,
        address_bits,
        phase,
        SUFFIX_CHUNKS,
    )
}

pub fn init_suffix_buckets_chunked(
    context: &CudaKernelContext,
    rows: &DeviceRows,
    u_evals: &DeviceFrVec,
    present: &[LookupTableKind<RISCV_XLEN>],
    address_bits: usize,
    phase: usize,
    chunks: usize,
) -> Result<Vec<Vec<DeviceFrVec>>, CudaError> {
    if chunks == 0 {
        return Err(CudaError::InvariantViolation {
            reason: "a chunked suffix reduce needs at least one chunk",
        });
    }
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
    let _ = builder.arg(rows.lookup_index());
    let _ = builder.arg(&rows.table_index);
    let _ = builder.arg(&device_slots);
    let _ = builder.arg(&table_count_arg);
    let _ = builder.arg(&suffix_len_arg);
    let _ = builder.arg(&mut keys);
    let _ = builder.arg(&count);
    // SAFETY: thread `j < cycles` reads `lookup_index[2j]`/`[2j+1]` of
    // `2 * cycles`, `table_index[j]` of `cycles`, and `table_slots[table]` of
    // `table_count` only after bounds-checking `table` against it. Writes only
    // `keys[j]` of `cycles`.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    context.stream().synchronize()?;

    let bucket_count = present.len() * CHUNK_SIZE;
    let segments = segment_rows(context, &keys, rows.cycles, bucket_count)?;

    let device_suffix_ids = context.upload_u32_slice(&suffix_ids)?;
    let device_suffix_offsets = context.upload_u32_slice(&suffix_offsets)?;
    let device_suffix_counts = context.upload_u32_slice(&suffix_counts)?;

    let groups = suffix_ids.len() * CHUNK_SIZE;
    let mut slots = alloc_slots(context, groups * chunks)?;
    let blocks = CudaKernelContext::count_of(bucket_count)?;
    let chunks_arg = CudaKernelContext::count_of(chunks)?;
    let mut builder = context.stream().launch_builder(context.ap_suffix_reduce());
    let _ = builder.arg(&segments.order);
    let _ = builder.arg(&segments.offsets);
    let _ = builder.arg(&segments.counts);
    let _ = builder.arg(rows.lookup_index());
    let _ = builder.arg(u_evals.limbs());
    let _ = builder.arg(&device_suffix_ids);
    let _ = builder.arg(&device_suffix_offsets);
    let _ = builder.arg(&device_suffix_counts);
    let _ = builder.arg(&suffix_len_arg);
    let _ = builder.arg(&mut slots);
    // SAFETY: block `blockIdx.x < present.len() * CHUNK_SIZE` gives
    // `slot = blockIdx.x / CHUNK_SIZE < present.len()`, bounding its
    // `suffix_offsets`/`suffix_counts` reads, and `suffix_ids[family_base + s]`
    // for `s < suffix_counts[slot]` stays inside `suffix_ids`. Its segment of
    // `order` holds row indices `< cycles`, bounding the `lookup_index` and
    // `u_evals` reads. Thread 0 writes the `2 * ACCUM_LIMBS` lanes at
    // `slots[((family_base + s) * CHUNK_SIZE + bucket) * 2 * ACCUM_LIMBS]`, inside
    // the `suffix_ids.len() * CHUNK_SIZE` folded slots `alloc_slots` reserved;
    // distinct blocks have distinct `(slot, bucket)`, so targets are distinct.
    // `acc` is indexed by `s < families <= MAX_SUFFIXES`, its declared extent.
    // `blockIdx.y < chunks` selects a stride-`gridDim.y * blockDim.x` subsequence
    // of that segment and the matching chunk slot, so the `groups * chunks` slots
    // reserved above hold one disjoint target per (family, bucket, chunk); blocks
    // whose chunk starts past `counts[blockIdx.x]` return before any access, and
    // their slots stay at the zero `alloc_slots` left, which is the identity the
    // chunk fold adds.
    // Shared memory is `BLOCK * 2 * ACCUM_LIMBS` u64s — the folded accumulator
    // width the reduction tree operates on — matching `shared_mem_bytes`.
    let _ = unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (blocks, chunks_arg, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: BLOCK * 2 * ACCUM_LIMBS as u32 * size_of::<u64>() as u32,
        })
    }?;
    context.stream().synchronize()?;

    let folded = if chunks == 1 {
        slots
    } else {
        fold_slot_chunks(context, &slots, groups, chunks)?
    };
    let buckets = finalize_slots(context, &folded, groups)?;

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

#[tracing::instrument(skip_all, name = "ap_condense_u_evals")]
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
    let _ = builder.arg(rows.lookup_index());
    let _ = builder.arg(u_evals.limbs_mut());
    let _ = builder.arg(v_prev.limbs());
    let _ = builder.arg(&suffix_len_arg);
    let _ = builder.arg(&count);
    // SAFETY: thread `j < cycles` reads `lookup_index[2j]`/`[2j+1]` of
    // `2 * cycles` and `v_prev[chunk]` with `chunk` masked to `CHUNK_SIZE - 1`
    // and `v_prev` length-checked above, then read-modify-writes exactly
    // `u_evals[j]` of `cycles`. In-place but one thread per element, so no
    // thread reads another's target.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    context.stream().synchronize()?;
    Ok(())
}

pub fn flag_claims(
    context: &CudaKernelContext,
    rows: &DeviceRows,
    eq_cycle: &DeviceFrVec,
    table_count: usize,
) -> Result<(Vec<Fr>, Fr), CudaError> {
    if eq_cycle.len() != rows.cycles() {
        return Err(CudaError::LengthMismatch {
            expected: rows.cycles(),
            got: eq_cycle.len(),
        });
    }
    let count = CudaKernelContext::count_of(rows.cycles())?;
    let table_count_arg = CudaKernelContext::count_of(table_count)?;

    let mut keys = context.alloc_u32(rows.cycles())?;
    let mut builder = context.stream().launch_builder(context.ap_flag_keys());
    let _ = builder.arg(rows.table_index());
    let _ = builder.arg(&table_count_arg);
    let _ = builder.arg(&mut keys);
    let _ = builder.arg(&count);
    // SAFETY: thread `j < cycles` reads `table_index[j]` of `cycles` and writes
    // only `keys[j]` of `cycles`, either `SKIP` or a value `< table_count`.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    context.stream().synchronize()?;

    let segments = segment_rows(context, &keys, rows.cycles(), table_count)?;
    let mut sums = context.alloc(table_count)?;
    let blocks = CudaKernelContext::count_of(table_count)?;
    let mut builder = context.stream().launch_builder(context.ap_flag_sums());
    let _ = builder.arg(&segments.order);
    let _ = builder.arg(&segments.offsets);
    let _ = builder.arg(&segments.counts);
    let _ = builder.arg(eq_cycle.limbs());
    let _ = builder.arg(sums.limbs_mut());
    // SAFETY: block `bucket < table_count` reads `order[offsets[bucket] ..
    // + counts[bucket]]`, whose elements are row indices `< cycles`, bounding the
    // `eq_cycle` reads. Thread 0 writes only `out[bucket]` of `table_count`.
    // Shared memory is `BLOCK * LIMBS` u64s, matching `shared_mem_bytes`.
    let _ = unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
        })
    }?;
    context.stream().synchronize()?;

    let raf_blocks = count.div_ceil(BLOCK).max(1);
    let mut partials = context.alloc(raf_blocks as usize)?;
    let mut builder = context.stream().launch_builder(context.ap_raf_flag_sum());
    let _ = builder.arg(rows.raf_flag());
    let _ = builder.arg(eq_cycle.limbs());
    let _ = builder.arg(partials.limbs_mut());
    let _ = builder.arg(&count);
    // SAFETY: thread `j < cycles` reads `raf_flag[j]` and `eq_cycle[j]`, both of
    // `cycles`; lanes with `j >= cycles` seed the tree with field zero. Thread 0
    // writes only `partials[blockIdx.x]` of `raf_blocks`. Shared memory is
    // `BLOCK * LIMBS` u64s, matching `shared_mem_bytes`.
    let _ = unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (raf_blocks, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
        })
    }?;
    context.stream().synchronize()?;

    Ok((sums.to_host()?, context.sum(&partials)?))
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

    use super::{
        condense_u_evals, init_raf_buckets, init_raf_buckets_chunked, init_suffix_buckets_chunked,
        DeviceRows, CHUNK_LEN, CHUNK_SIZE,
    };
    use crate::cuda::common::context::{shared_context, CudaKernelContext};
    use crate::cuda::common::testing::fr;
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
        fn chunked_raf_buckets_match_the_reference_init_phase(
            log_t in 4usize..=10,
            seed in any::<u64>(),
            phase in 0usize..PHASES,
            chunks in prop::sample::select(vec![1usize, 2, 3, 8, 64]),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let host = reference_at_phase(log_t, seed, phase);
            let rows = device_rows(context, host.rows());
            let u_evals = context.upload(host.u_evals()).expect("upload u_evals");

            let got = init_raf_buckets_chunked(
                context, &rows, &u_evals, ADDRESS_BITS, phase, chunks,
            )
            .expect("device init_raf_buckets_chunked");

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
                    "{} bucket diverged at phase {} with {} chunks",
                    label,
                    phase,
                    chunks
                );
            }

            if CANONICAL_INSTRUCTION_ADDRESS {
                prop_assert_eq!(
                    &got.upper_all_ones.to_host().expect("download"),
                    &evals(&host.raf_upper_all_ones.q_shift),
                    "upper_all_ones bucket diverged at phase {} with {} chunks",
                    phase,
                    chunks
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
        fn chunked_suffix_buckets_match_the_reference_init_phase(
            log_t in 4usize..=10,
            seed in any::<u64>(),
            phase in 0usize..PHASES,
            chunks in prop::sample::select(vec![1usize, 2, 3, 8, 64]),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let host = reference_at_phase(log_t, seed, phase);
            let rows = device_rows(context, host.rows());
            let u_evals = context.upload(host.u_evals()).expect("upload u_evals");
            let present: Vec<LookupTableKind<RISCV_XLEN>> =
                host.suffix_tables.iter().map(|(table, _)| *table).collect();

            let got = init_suffix_buckets_chunked(
                context, &rows, &u_evals, &present, ADDRESS_BITS, phase, chunks,
            )
            .expect("device init_suffix_buckets_chunked");

            prop_assert_eq!(got.len(), host.suffix_tables.len());
            for (columns, (table, expected)) in got.iter().zip(&host.suffix_tables) {
                prop_assert_eq!(columns.len(), expected.len());
                for (slot, (column, want)) in columns.iter().zip(expected).enumerate() {
                    prop_assert_eq!(
                        &column.to_host().expect("download"),
                        &evals(want),
                        "table {:?} suffix slot {} diverged at phase {} with {} chunks",
                        table,
                        slot,
                        phase,
                        chunks
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

    proptest! {
        #[test]
        fn flag_claims_match_the_reference_output_claims(
            log_t in 4usize..=10,
            seed in any::<u64>(),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let host = reference_at_phase(log_t, seed, 0);
            let device = device_rows(context, host.rows());

            let r_cycle: Vec<Fr> = (0..log_t).map(|i| fr(seed + i as u64 + 301)).collect();
            let eq_cycle = crate::reference::views::eq_table(&r_cycle);
            let uploaded = context.upload(&eq_cycle).expect("upload eq_cycle");

            let table_count = LookupTableKind::<RISCV_XLEN>::COUNT;
            let (flags, raf_flag) =
                super::flag_claims(context, &device, &uploaded, table_count)
                    .expect("device flag_claims");

            let mut expected_flags = vec![Fr::from(0u64); table_count];
            let mut expected_raf = Fr::from(0u64);
            for (row, &eq) in host.rows().iter().zip(&eq_cycle) {
                if let Some(index) = row.table_index.0 {
                    expected_flags[index] += eq;
                }
                if row.raf_flag.0 {
                    expected_raf += eq;
                }
            }
            prop_assert_eq!(flags, expected_flags, "lookup table flag claims diverged");
            prop_assert_eq!(raf_flag, expected_raf, "instruction RAF flag claim diverged");
        }
    }

    #[test]
    fn cycle_window_bucket_sums_match_the_whole_domain() {
        let Some(context) = shared_context() else {
            return;
        };
        for phase in [0usize, 1, 3] {
            let host = reference_at_phase(8, 11, phase);
            let all = host.rows();
            let u_all = host.u_evals();
            let whole_rows = device_rows(context, all);
            let whole_u = context.upload(u_all).expect("upload u_evals");
            let whole = init_raf_buckets(context, &whole_rows, &whole_u, ADDRESS_BITS, phase)
                .expect("whole-domain raf buckets");

            let half = all.len() / 2;
            let mut summed: Vec<Vec<Fr>> = Vec::new();
            for window in [0..half, half..all.len()] {
                let rows = device_rows(context, &all[window.clone()]);
                let u_evals = context
                    .upload(&u_all[window])
                    .expect("upload window u_evals");
                let part = init_raf_buckets(context, &rows, &u_evals, ADDRESS_BITS, phase)
                    .expect("window raf buckets");
                let columns = [
                    &part.shift_half,
                    &part.shift_full,
                    &part.left,
                    &part.right,
                    &part.identity,
                    &part.upper_all_ones,
                ]
                .map(|column| column.to_host().expect("download"));
                if summed.is_empty() {
                    summed = columns.to_vec();
                } else {
                    for (target, addend) in summed.iter_mut().zip(columns.iter()) {
                        for (slot, value) in target.iter_mut().zip(addend) {
                            *slot += *value;
                        }
                    }
                }
            }

            for (label, device, expected) in [
                ("shift_half", &whole.shift_half, &summed[0]),
                ("shift_full", &whole.shift_full, &summed[1]),
                ("left", &whole.left, &summed[2]),
                ("right", &whole.right, &summed[3]),
                ("identity", &whole.identity, &summed[4]),
                ("upper_all_ones", &whole.upper_all_ones, &summed[5]),
            ] {
                let got = device.to_host().expect("download");
                assert_eq!(
                    &got, expected,
                    "phase {phase}: the {label} raf bucket accumulator over the whole cycle \
                     domain must equal the sum of its two cycle windows — every bucket is a sum \
                     over the cycles that land in it, which is what makes a cycle-range split \
                     across devices exact",
                );
            }
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
