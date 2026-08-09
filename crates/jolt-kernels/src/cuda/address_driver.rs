#![expect(
    dead_code,
    reason = "implementation target: the instruction read-RAF kernel wires this once it lands"
)]

use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_claims::protocols::jolt::geometry::instruction::CANONICAL_INSTRUCTION_ADDRESS;
use jolt_field::{Field, Fr};
use jolt_lookup_tables::tables::LookupTableKind;
use jolt_lookup_tables::XLEN as RISCV_XLEN;

use super::address_phase::{
    condense_u_evals, init_raf_buckets, init_suffix_buckets, DeviceRows, RafBuckets, CHUNK_LEN,
    CHUNK_SIZE,
};
use super::combine::{combine_terms, CombineTerm};
use super::context::{CudaKernelContext, BLOCK};
use super::device::{require_fr_slice, DeviceFrVec, LIMBS};
use super::error::CudaError;
use super::prefixes::{default_checkpoints, NUM_PREFIXES};
use super::unreduced::{alloc_slots, finalize_slots, ACCUM_LIMBS};

const RAF_TERMS: usize = 3;
const HINT_POINTS: usize = 2;
const RAF_CHECKPOINTS: usize = 4;
const NO_PREFIX: u32 = u32::MAX;

pub struct DeviceAddressPhase {
    rows: DeviceRows,
    u_evals: DeviceFrVec,
    present: Vec<LookupTableKind<RISCV_XLEN>>,
    layout: TermLayout,
    address_bits: usize,
    phase: usize,
    rounds_bound: usize,
    phase_challenges: Vec<Fr>,
    checkpoints: DeviceFrVec,
    raf_checkpoints: DeviceFrVec,
    v_tables: Vec<DeviceFrVec>,
    tables: PhaseTables,
}

struct PhaseTables {
    prefixes: DeviceFrVec,
    suffixes: DeviceFrVec,
    raf_prefix: DeviceFrVec,
    raf: RafBuckets,
    columns: usize,
    len: usize,
}

struct TermLayout {
    scales: Vec<u32>,
    prefix_ids: Vec<u32>,
    suffix_slots: Vec<u32>,
    offsets: Vec<u32>,
    counts: Vec<u32>,
    suffix_bases: Vec<u32>,
}

fn term_layout(present: &[LookupTableKind<RISCV_XLEN>]) -> Result<TermLayout, CudaError> {
    let mut layout = TermLayout {
        scales: Vec::new(),
        prefix_ids: Vec::new(),
        suffix_slots: Vec::new(),
        offsets: Vec::new(),
        counts: Vec::new(),
        suffix_bases: Vec::new(),
    };
    let mut base = 0u32;
    for table in present {
        let terms: Vec<CombineTerm> = combine_terms(*table)?;
        layout.offsets.push(layout.scales.len() as u32);
        layout.counts.push(terms.len() as u32);
        layout.suffix_bases.push(base);
        base += table.suffixes().len() as u32;
        for term in terms {
            layout.scales.push(term.scale as u32);
            layout
                .prefix_ids
                .push(term.prefix.map_or(NO_PREFIX, |prefix| prefix as u32));
            layout.suffix_slots.push(term.suffix as u32);
        }
    }
    Ok(layout)
}

impl DeviceAddressPhase {
    pub fn new<F: Field>(
        context: &CudaKernelContext,
        lookup_index: &[u128],
        table_index: &[Option<usize>],
        raf_flag: &[bool],
        r_reduction: &[F],
        address_bits: usize,
    ) -> Result<Self, CudaError> {
        if !address_bits.is_multiple_of(CHUNK_LEN) {
            return Err(CudaError::InvariantViolation {
                reason: "the device address phase supports only whole 8-variable phases",
            });
        }
        let rows = DeviceRows::new(context, lookup_index, table_index, raf_flag)?;
        let u_evals = context.eq_evals(require_fr_slice(r_reduction)?)?;
        if u_evals.len() != rows.cycles() {
            return Err(CudaError::LengthMismatch {
                expected: rows.cycles(),
                got: u_evals.len(),
            });
        }

        let mut used = [false; LookupTableKind::<RISCV_XLEN>::COUNT];
        for &index in table_index.iter().flatten() {
            *used.get_mut(index).ok_or(CudaError::InvariantViolation {
                reason: "a stage-5 row selects an unknown lookup table",
            })? = true;
        }
        let present: Vec<LookupTableKind<RISCV_XLEN>> = LookupTableKind::<RISCV_XLEN>::iter()
            .filter(|table| used[table.index()])
            .collect();
        let layout = term_layout(&present)?;

        let checkpoints = default_checkpoints(context)?;
        let mut raf_checkpoints = vec![Fr::from(0u64); RAF_CHECKPOINTS];
        if CANONICAL_INSTRUCTION_ADDRESS {
            raf_checkpoints[3] = Fr::from(1u64);
        }
        let raf_checkpoints = context.upload(&raf_checkpoints)?;

        let mut phase = Self {
            rows,
            u_evals,
            present,
            layout,
            address_bits,
            phase: 0,
            rounds_bound: 0,
            phase_challenges: Vec::with_capacity(CHUNK_LEN),
            checkpoints,
            raf_checkpoints,
            v_tables: Vec::new(),
            tables: PhaseTables::empty(context)?,
        };
        phase.init_phase(context, 0)?;
        Ok(phase)
    }

    fn phases(&self) -> usize {
        self.address_bits / CHUNK_LEN
    }

    fn suffix_len(&self, phase: usize) -> Result<usize, CudaError> {
        self.address_bits
            .checked_sub((phase + 1) * CHUNK_LEN)
            .ok_or(CudaError::InvariantViolation {
                reason: "address phase index exceeds the address width",
            })
    }

    fn init_phase(&mut self, context: &CudaKernelContext, phase: usize) -> Result<(), CudaError> {
        if phase != 0 {
            let v_prev = self
                .v_tables
                .last()
                .ok_or(CudaError::InvariantViolation {
                    reason: "a later address phase found no previous eq table",
                })?
                .try_clone()?;
            condense_u_evals(
                context,
                &self.rows,
                &mut self.u_evals,
                &v_prev,
                self.address_bits,
                phase,
            )?;
        }

        let raf = init_raf_buckets(context, &self.rows, &self.u_evals, self.address_bits, phase)?;
        let suffix_columns = init_suffix_buckets(
            context,
            &self.rows,
            &self.u_evals,
            &self.present,
            self.address_bits,
            phase,
        )?;
        let suffixes = flatten(context, suffix_columns)?;
        let suffix_count = suffixes.len() / CHUNK_SIZE;

        let suffix_len = self.suffix_len(phase)?;
        let prefixes = self.build_prefix_tables(context, suffix_len)?;
        let raf_prefix = self.build_raf_prefix_tables(context, phase)?;

        self.tables = PhaseTables {
            prefixes,
            suffixes,
            raf_prefix,
            raf,
            columns: suffix_count,
            len: CHUNK_SIZE,
        };
        self.phase = phase;
        self.phase_challenges.clear();
        Ok(())
    }

    fn build_prefix_tables(
        &self,
        context: &CudaKernelContext,
        suffix_len: usize,
    ) -> Result<DeviceFrVec, CudaError> {
        let mut out = context.alloc(NUM_PREFIXES * CHUNK_SIZE)?;
        let suffix_len = CudaKernelContext::count_of(suffix_len)?;
        let prefix_count = CudaKernelContext::count_of(NUM_PREFIXES)?;
        let chunk_count = CudaKernelContext::count_of(CHUNK_SIZE)?;
        let mut builder = context.stream().launch_builder(context.ap_prefix_tables());
        let _ = builder.arg(self.checkpoints.limbs());
        let _ = builder.arg(&suffix_len);
        let _ = builder.arg(out.limbs_mut());
        let _ = builder.arg(&prefix_count);
        // SAFETY: thread `x < CHUNK_SIZE` reads any of the `NUM_PREFIXES`
        // checkpoints (`self.checkpoints` is allocated at that length) and writes
        // `out[prefix * CHUNK_SIZE + x]` for `prefix < prefix_count`, one slot per
        // (prefix, x) of `NUM_PREFIXES * CHUNK_SIZE`.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(chunk_count)) }?;
        context.stream().synchronize()?;
        Ok(out)
    }

    fn build_raf_prefix_tables(
        &self,
        context: &CudaKernelContext,
        phase: usize,
    ) -> Result<DeviceFrVec, CudaError> {
        let chunk_upper_bits = (self.address_bits / 2)
            .saturating_sub(phase * CHUNK_LEN)
            .min(CHUNK_LEN);
        let mut out = context.alloc(RAF_CHECKPOINTS * CHUNK_SIZE)?;
        let chunk_upper_bits = CudaKernelContext::count_of(chunk_upper_bits)?;
        let canonical = u32::from(CANONICAL_INSTRUCTION_ADDRESS);
        let chunk_count = CudaKernelContext::count_of(CHUNK_SIZE)?;
        let mut builder = context.stream().launch_builder(context.ap_raf_prefix());
        let _ = builder.arg(self.raf_checkpoints.limbs());
        let _ = builder.arg(&chunk_upper_bits);
        let _ = builder.arg(&canonical);
        let _ = builder.arg(out.limbs_mut());
        // SAFETY: thread `x < CHUNK_SIZE` reads the four `raf_checkpoints`
        // elements (allocated at `RAF_CHECKPOINTS`) and writes
        // `out[lane * CHUNK_SIZE + x]` for `lane < RAF_CHECKPOINTS`, one slot per
        // (lane, x) of `RAF_CHECKPOINTS * CHUNK_SIZE`.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(chunk_count)) }?;
        context.stream().synchronize()?;
        Ok(out)
    }

    pub fn round_message_hinted(
        &self,
        context: &CudaKernelContext,
        gamma: Fr,
        previous_claim: Fr,
    ) -> Result<[Fr; 2], CudaError> {
        let _ = previous_claim;
        let half = self.tables.len / 2;
        if half == 0 {
            return Err(CudaError::InvariantViolation {
                reason: "the address round tables are already fully bound",
            });
        }

        let scales = context.upload_u32_slice(&self.layout.scales)?;
        let prefix_ids = context.upload_u32_slice(&self.layout.prefix_ids)?;
        let suffix_slots = context.upload_u32_slice(&self.layout.suffix_slots)?;
        let offsets = context.upload_u32_slice(&self.layout.offsets)?;
        let counts = context.upload_u32_slice(&self.layout.counts)?;
        let suffix_bases = context.upload_u32_slice(&self.layout.suffix_bases)?;

        let table_count = CudaKernelContext::count_of(self.present.len())?;
        let half_count = CudaKernelContext::count_of(half)?;
        let stride = CudaKernelContext::count_of(self.tables.len)?;
        let raf_count = CudaKernelContext::count_of(RAF_TERMS)?;
        let lanes = HINT_POINTS as u32 * (1 + RAF_TERMS) as u32;
        let blocks = half_count.div_ceil(BLOCK).max(1);
        let mut slots = alloc_slots(context, lanes as usize * blocks as usize)?;

        let mut builder = context
            .stream()
            .launch_builder(context.ap_round_message_hinted());
        let _ = builder.arg(self.tables.prefixes.limbs());
        let _ = builder.arg(&prefix_ids);
        let _ = builder.arg(&suffix_slots);
        let _ = builder.arg(&scales);
        let _ = builder.arg(&offsets);
        let _ = builder.arg(&counts);
        let _ = builder.arg(self.tables.suffixes.limbs());
        let _ = builder.arg(&suffix_bases);
        let _ = builder.arg(&table_count);
        let _ = builder.arg(self.tables.raf_prefix.limbs());
        let _ = builder.arg(self.tables.raf.shift_half.limbs());
        let _ = builder.arg(self.tables.raf.shift_full.limbs());
        let _ = builder.arg(self.tables.raf.left.limbs());
        let _ = builder.arg(self.tables.raf.right.limbs());
        let _ = builder.arg(self.tables.raf.identity.limbs());
        let _ = builder.arg(&raf_count);
        let _ = builder.arg(&stride);
        let _ = builder.arg(&half_count);
        let _ = builder.arg(&mut slots);
        // SAFETY: identical indexing to `ap_round_message` — thread `b < half`
        // reads index `b` and `b + half` of columns that are all `stride =
        // tables.len >= 2 * half` elements long, and term indices come from
        // `term_layout` (`NO_PREFIX` is never dereferenced,
        // `suffix_bases[t] + suffix_slots[term]` indexes a live suffix column).
        // Thread 0 writes the `2 * ACCUM_LIMBS` lanes at
        // `slots[(lane * gridDim.x + blockIdx.x) * 2 * ACCUM_LIMBS]` for
        // `lane < lanes`, inside the `lanes * blocks` folded slots `alloc_slots`
        // reserved. Shared memory is `BLOCK * 2 * ACCUM_LIMBS` u64s — the folded
        // accumulator width the reduction tree operates on — matching
        // `shared_mem_bytes`.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * 2 * ACCUM_LIMBS as u32 * size_of::<u64>() as u32,
            })
        }?;
        context.stream().synchronize()?;

        let partials = finalize_slots(context, &slots, lanes as usize * blocks as usize)?;
        let totals = reduce_lanes(context, partials, lanes, blocks)?.to_host()?;
        let gamma_sqr = gamma * gamma;
        let mut evals = [Fr::from(0u64); HINT_POINTS];
        for (point, eval) in evals.iter_mut().enumerate() {
            let base = point * (1 + RAF_TERMS);
            let read = totals[base];
            let left = totals[base + 1];
            let right = totals[base + 2];
            let identity = totals[base + 3];
            *eval = read + gamma * left + gamma_sqr * (right + identity);
        }
        Ok(evals)
    }

    pub fn round_message(
        &self,
        context: &CudaKernelContext,
        gamma: Fr,
    ) -> Result<[Fr; 3], CudaError> {
        let half = self.tables.len / 2;
        if half == 0 {
            return Err(CudaError::InvariantViolation {
                reason: "the address round tables are already fully bound",
            });
        }

        let scales = context.upload_u32_slice(&self.layout.scales)?;
        let prefix_ids = context.upload_u32_slice(&self.layout.prefix_ids)?;
        let suffix_slots = context.upload_u32_slice(&self.layout.suffix_slots)?;
        let offsets = context.upload_u32_slice(&self.layout.offsets)?;
        let counts = context.upload_u32_slice(&self.layout.counts)?;
        let suffix_bases = context.upload_u32_slice(&self.layout.suffix_bases)?;

        let table_count = CudaKernelContext::count_of(self.present.len())?;
        let half_count = CudaKernelContext::count_of(half)?;
        let stride = CudaKernelContext::count_of(self.tables.len)?;
        let raf_count = CudaKernelContext::count_of(RAF_TERMS)?;
        let lanes = 3 * (1 + RAF_TERMS) as u32;
        let blocks = half_count.div_ceil(BLOCK).max(1);
        let mut partials = context.alloc(lanes as usize * blocks as usize)?;

        let mut builder = context.stream().launch_builder(context.ap_round_message());
        let _ = builder.arg(self.tables.prefixes.limbs());
        let _ = builder.arg(&prefix_ids);
        let _ = builder.arg(&suffix_slots);
        let _ = builder.arg(&scales);
        let _ = builder.arg(&offsets);
        let _ = builder.arg(&counts);
        let _ = builder.arg(self.tables.suffixes.limbs());
        let _ = builder.arg(&suffix_bases);
        let _ = builder.arg(&table_count);
        let _ = builder.arg(self.tables.raf_prefix.limbs());
        let _ = builder.arg(self.tables.raf.shift_half.limbs());
        let _ = builder.arg(self.tables.raf.shift_full.limbs());
        let _ = builder.arg(self.tables.raf.left.limbs());
        let _ = builder.arg(self.tables.raf.right.limbs());
        let _ = builder.arg(self.tables.raf.identity.limbs());
        let _ = builder.arg(&raf_count);
        let _ = builder.arg(&stride);
        let _ = builder.arg(&half_count);
        let _ = builder.arg(partials.limbs_mut());
        // SAFETY: thread `b < half` reads index `b` and `b + half` of each
        // column, and every column here is `stride = tables.len >= 2 * half`
        // elements long: the prefix and RAF-prefix buffers are strided by
        // `CHUNK_SIZE`, the flattened suffix buffer holds `columns * CHUNK_SIZE`,
        // and the five RAF bucket lanes hold `CHUNK_SIZE` each. Term indices come
        // from `term_layout`: `prefix_ids` are `Prefixes` discriminants
        // (`NO_PREFIX` is never dereferenced) and
        // `suffix_bases[t] + suffix_slots[term]` indexes a live suffix column.
        // Thread 0 writes `partials[lane * gridDim.x + blockIdx.x]` of
        // `lanes * blocks`. Shared memory is `BLOCK * LIMBS` u64s, matching
        // `shared_mem_bytes`.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
            })
        }?;
        context.stream().synchronize()?;

        let totals = reduce_lanes(context, partials, lanes, blocks)?.to_host()?;
        let gamma_sqr = gamma * gamma;
        let mut evals = [Fr::from(0u64); 3];
        for (c, eval) in evals.iter_mut().enumerate() {
            let base = c * (1 + RAF_TERMS);
            let read = totals[base];
            let left = totals[base + 1];
            let right = totals[base + 2];
            let identity = totals[base + 3];
            *eval = read + gamma * left + gamma_sqr * (right + identity);
        }
        Ok(evals)
    }

    pub fn bind(&mut self, context: &CudaKernelContext, challenge: Fr) -> Result<(), CudaError> {
        self.tables.bind(context, challenge)?;
        self.phase_challenges.push(challenge);
        self.rounds_bound += 1;

        if self.phase_challenges.len() == CHUNK_LEN {
            self.v_tables
                .push(context.eq_evals(&self.phase_challenges)?);
            self.checkpoints = self.tables.prefixes.try_clone()?;
            self.raf_checkpoints = self.tables.raf_prefix.try_clone()?;
            let next = self.phase + 1;
            if next < self.phases() {
                self.init_phase(context, next)?;
            }
        }
        Ok(())
    }

    pub fn checkpoints(&self, context: &CudaKernelContext) -> Result<Vec<Fr>, CudaError> {
        let _ = context;
        self.checkpoints.to_host()
    }

    pub fn raf_checkpoints(&self, context: &CudaKernelContext) -> Result<Vec<Fr>, CudaError> {
        let _ = context;
        self.raf_checkpoints.to_host()
    }

    pub const fn rows(&self) -> &DeviceRows {
        &self.rows
    }

    pub fn v_tables(&self) -> &[DeviceFrVec] {
        &self.v_tables
    }

    pub const fn rounds_bound(&self) -> usize {
        self.rounds_bound
    }
}

impl PhaseTables {
    fn empty(context: &CudaKernelContext) -> Result<Self, CudaError> {
        Ok(Self {
            prefixes: context.alloc(0)?,
            suffixes: context.alloc(0)?,
            raf_prefix: context.alloc(0)?,
            raf: RafBuckets {
                shift_half: context.alloc(0)?,
                shift_full: context.alloc(0)?,
                left: context.alloc(0)?,
                right: context.alloc(0)?,
                identity: context.alloc(0)?,
                upper_all_ones: context.alloc(0)?,
            },
            columns: 0,
            len: 0,
        })
    }

    fn bind(&mut self, context: &CudaKernelContext, challenge: Fr) -> Result<(), CudaError> {
        self.prefixes = bind_strided(context, &self.prefixes, self.len, challenge)?;
        self.suffixes = bind_strided(context, &self.suffixes, self.len, challenge)?;
        self.raf_prefix = bind_strided(context, &self.raf_prefix, self.len, challenge)?;
        for lane in [
            &mut self.raf.shift_half,
            &mut self.raf.shift_full,
            &mut self.raf.left,
            &mut self.raf.right,
            &mut self.raf.identity,
            &mut self.raf.upper_all_ones,
        ] {
            *lane = bind_strided(context, lane, self.len, challenge)?;
        }
        self.len /= 2;
        Ok(())
    }
}

fn bind_strided(
    context: &CudaKernelContext,
    values: &DeviceFrVec,
    stride: usize,
    challenge: Fr,
) -> Result<DeviceFrVec, CudaError> {
    if values.is_empty() || stride < 2 {
        return context.alloc(0);
    }
    let columns = values.len() / stride;
    let half = stride / 2;
    let mut out = context.alloc(columns * half)?;
    let challenge = context.upload(&[challenge])?;
    let count = CudaKernelContext::count_of(columns * half)?;
    let half_arg = CudaKernelContext::count_of(half)?;
    let stride_arg = CudaKernelContext::count_of(stride)?;
    let mut builder = context.stream().launch_builder(context.ap_bind_strided());
    let _ = builder.arg(values.limbs());
    let _ = builder.arg(challenge.limbs());
    let _ = builder.arg(out.limbs_mut());
    let _ = builder.arg(&half_arg);
    let _ = builder.arg(&stride_arg);
    let _ = builder.arg(&count);
    // SAFETY: thread `i < columns * half` splits into `column = i / half` and
    // `b = i % half`, reading `in[column * stride + b]` and
    // `in[column * stride + b + half]` — both inside `in`'s `columns * stride`
    // elements — plus the single-element challenge buffer, and writing only
    // `out[i]` of `columns * half`. `out` is a fresh allocation distinct from
    // `in`.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    context.stream().synchronize()?;
    Ok(out)
}

fn flatten(
    context: &CudaKernelContext,
    columns: Vec<Vec<DeviceFrVec>>,
) -> Result<DeviceFrVec, CudaError> {
    let flat: Vec<DeviceFrVec> = columns.into_iter().flatten().collect();
    if flat.is_empty() {
        return context.alloc(0);
    }
    let mut out = context.alloc(flat.len() * CHUNK_SIZE)?;
    for (index, column) in flat.iter().enumerate() {
        context.copy_into(&mut out, index * CHUNK_SIZE, column)?;
    }
    Ok(out)
}

fn reduce_lanes(
    context: &CudaKernelContext,
    mut partials: DeviceFrVec,
    lanes: u32,
    mut width: u32,
) -> Result<DeviceFrVec, CudaError> {
    while width > 1 {
        let next = width.div_ceil(2);
        let mut folded = context.alloc(lanes as usize * next as usize)?;
        let mut builder = context.stream().launch_builder(context.lane_sum_reduce());
        let _ = builder.arg(partials.limbs());
        let _ = builder.arg(folded.limbs_mut());
        let _ = builder.arg(&lanes);
        let _ = builder.arg(&width);
        let _ = builder.arg(&next);
        // SAFETY: thread `(i < next, lane < lanes)` reads `in[lane * width + i]`
        // and, when `i + next < width`, its mate at `+ next` — both inside `in`'s
        // `lanes * width` elements — and writes only `out[lane * next + i]` of
        // `lanes * next`. Index sets are pairwise disjoint and `out` is a
        // distinct allocation.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (next.div_ceil(BLOCK), lanes, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        context.stream().synchronize()?;
        partials = folded;
        width = next;
    }
    Ok(partials)
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
    use jolt_field::Fr;
    use jolt_lookup_tables::tables::LookupTableKind;
    use jolt_lookup_tables::XLEN as RISCV_XLEN;
    use jolt_poly::UnivariatePoly;
    use jolt_witness::witnesses::{InstructionRafFlag, LookupIndex, TableIndex};
    use proptest::prelude::*;
    use std::num::NonZeroUsize;

    use super::super::context::shared_context;
    use super::super::testing::fr;
    use super::DeviceAddressPhase;
    use crate::reference::instruction_read_raf::{
        InstructionReadRafKernel, InstructionReadRafWitness,
    };

    const ADDRESS_BITS: usize = 128;

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

    proptest! {
        #[test]
        fn address_rounds_match_the_reference_round_for_round(
            log_t in 4usize..=8,
            seed in any::<u64>(),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let jolt_rows = rows(log_t, seed);
            let gamma = fr(seed + 1);
            let r_reduction: Vec<Fr> = (0..log_t).map(|i| fr(seed + i as u64 + 3)).collect();
            let dimensions = InstructionReadRafDimensions::new(
                log_t,
                ADDRESS_BITS,
                NonZeroUsize::new(8).unwrap(),
            );

            let mut host = InstructionReadRafKernel::new(
                dimensions,
                &r_reduction,
                jolt_rows.clone(),
                gamma,
            )
            .expect("reference kernel");

            let indices: Vec<u128> = jolt_rows.iter().map(|row| row.lookup_index.0).collect();
            let tables: Vec<Option<usize>> =
                jolt_rows.iter().map(|row| row.table_index.0).collect();
            let flags: Vec<bool> = jolt_rows.iter().map(|row| row.raf_flag.0).collect();
            let mut device = DeviceAddressPhase::new(
                context,
                &indices,
                &tables,
                &flags,
                &r_reduction,
                ADDRESS_BITS,
            )
            .expect("device address phase");

            for round in 0..ADDRESS_BITS {
                let expected = host.address_message();
                let got = device.round_message(context, gamma).expect("device message");
                prop_assert_eq!(
                    got,
                    expected,
                    "address round {} message diverged",
                    round
                );

                let challenge = fr(seed + round as u64 + 71);
                host.bind(challenge).expect("reference bind");
                device.bind(context, challenge).expect("device bind");
            }

            let expected: Vec<Fr> = host
                .prefix_checkpoints()
                .iter()
                .map(|checkpoint| checkpoint.value())
                .collect();
            prop_assert_eq!(
                device.checkpoints(context).expect("checkpoints"),
                expected,
                "prefix checkpoints diverged after the address phase"
            );
        }

        #[test]
        fn hinted_address_rounds_match_the_reference_polynomial(
            log_t in 4usize..=8,
            seed in any::<u64>(),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let jolt_rows = rows(log_t, seed);
            let gamma = fr(seed + 1);
            let r_reduction: Vec<Fr> = (0..log_t).map(|i| fr(seed + i as u64 + 3)).collect();
            let dimensions = InstructionReadRafDimensions::new(
                log_t,
                ADDRESS_BITS,
                NonZeroUsize::new(8).unwrap(),
            );

            let mut host = InstructionReadRafKernel::new(
                dimensions,
                &r_reduction,
                jolt_rows.clone(),
                gamma,
            )
            .expect("reference kernel");

            let indices: Vec<u128> = jolt_rows.iter().map(|row| row.lookup_index.0).collect();
            let tables: Vec<Option<usize>> =
                jolt_rows.iter().map(|row| row.table_index.0).collect();
            let flags: Vec<bool> = jolt_rows.iter().map(|row| row.raf_flag.0).collect();
            let mut device = DeviceAddressPhase::new(
                context,
                &indices,
                &tables,
                &flags,
                &r_reduction,
                ADDRESS_BITS,
            )
            .expect("device address phase");

            for round in 0..ADDRESS_BITS {
                let reference = host.address_message();
                let expected = UnivariatePoly::from_evals(&reference);
                let previous_claim = reference[0] + reference[1];
                let got = UnivariatePoly::from_evals_and_hint(
                    previous_claim,
                    &device
                        .round_message_hinted(context, gamma, previous_claim)
                        .expect("device hinted message"),
                );
                prop_assert_eq!(
                    got.coefficients(),
                    expected.coefficients(),
                    "hinted address round {} polynomial diverged",
                    round
                );

                let challenge = fr(seed + round as u64 + 71);
                host.bind(challenge).expect("reference bind");
                device.bind(context, challenge).expect("device bind");
            }
        }
    }
}
