#![expect(
    dead_code,
    reason = "implementation target: the instruction read-RAF kernel wires this once it lands"
)]

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use jolt_claims::protocols::jolt::geometry::instruction::CANONICAL_INSTRUCTION_ADDRESS;
use jolt_field::{Field, Fr};
use jolt_lookup_tables::tables::LookupTableKind;
use jolt_lookup_tables::XLEN as RISCV_XLEN;

use super::address_phase::{
    condense_u_evals, init_raf_buckets, init_raf_columns, init_suffix_columns, DeviceRows,
    RafBuckets, CHUNK_LEN, CHUNK_SIZE, RAF_LANES,
};
use super::combine::{combine_terms, CombineTerm};
use super::prefixes::{default_checkpoints, prefix_mle_round, update_checkpoints, HINT_POINTS};
use crate::cuda::common::context::{context_for, CudaKernelContext, BLOCK};
use crate::cuda::common::device::{fr_limbs, require_fr_slice, DeviceFrVec};
use crate::cuda::common::devices::{fan_out, DeviceTask};
use crate::cuda::common::error::CudaError;
use crate::cuda::common::primitives::reduce_lanes;
use crate::cuda::common::unreduced::{alloc_slots, finalize_slots, ACCUM_LIMBS};

const RAF_TERMS: usize = 3;
const RAF_CHECKPOINTS: usize = 4;
const NO_PREFIX: u32 = u32::MAX;

pub struct AddressShard {
    pub ordinal: usize,
    pub rows: std::sync::Arc<DeviceRows>,
    pub u_evals: DeviceFrVec,
}

pub struct DeviceAddressPhase {
    shards: Vec<AddressShard>,
    present: Vec<LookupTableKind<RISCV_XLEN>>,
    layout: DeviceTermLayout,
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

struct DeviceTermLayout {
    ordinal: usize,
    scales: CudaSlice<u32>,
    prefix_ids: CudaSlice<u32>,
    suffix_slots: CudaSlice<u32>,
    offsets: CudaSlice<u32>,
    counts: CudaSlice<u32>,
    suffix_bases: CudaSlice<u32>,
}

impl TermLayout {
    fn upload(&self, context: &CudaKernelContext) -> Result<DeviceTermLayout, CudaError> {
        Ok(DeviceTermLayout {
            ordinal: context.ordinal(),
            scales: context.upload_u32_slice(&self.scales)?,
            prefix_ids: context.upload_u32_slice(&self.prefix_ids)?,
            suffix_slots: context.upload_u32_slice(&self.suffix_slots)?,
            offsets: context.upload_u32_slice(&self.offsets)?,
            counts: context.upload_u32_slice(&self.counts)?,
            suffix_bases: context.upload_u32_slice(&self.suffix_bases)?,
        })
    }
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
        let rows = std::sync::Arc::new(DeviceRows::new(
            context,
            lookup_index,
            table_index,
            raf_flag,
        )?);
        let mut used = [false; LookupTableKind::<RISCV_XLEN>::COUNT];
        for slot in table_index {
            if let Some(index) = *slot {
                *used.get_mut(index).ok_or(CudaError::InvariantViolation {
                    reason: "a stage-5 row selects an unknown lookup table",
                })? = true;
            }
        }
        let u_evals = context.eq_evals(require_fr_slice(r_reduction)?)?;
        let shards = vec![AddressShard {
            ordinal: context.ordinal(),
            rows,
            u_evals,
        }];
        Self::with_shards(context, shards, &used, address_bits)
    }

    pub fn with_shards(
        context: &CudaKernelContext,
        shards: Vec<AddressShard>,
        used: &[bool; LookupTableKind::<RISCV_XLEN>::COUNT],
        address_bits: usize,
    ) -> Result<Self, CudaError> {
        if !address_bits.is_multiple_of(CHUNK_LEN) {
            return Err(CudaError::InvariantViolation {
                reason: "the device address phase supports only whole 8-variable phases",
            });
        }
        if shards.is_empty() {
            return Err(CudaError::InvariantViolation {
                reason: "the device address phase was given no cycle window",
            });
        }
        for shard in &shards {
            if shard.u_evals.len() != shard.rows.cycles() {
                return Err(CudaError::LengthMismatch {
                    expected: shard.rows.cycles(),
                    got: shard.u_evals.len(),
                });
            }
        }

        let present: Vec<LookupTableKind<RISCV_XLEN>> = LookupTableKind::<RISCV_XLEN>::iter()
            .filter(|table| used[table.index()])
            .collect();
        let layout = term_layout(&present)?.upload(context)?;

        let checkpoints = default_checkpoints(context)?;
        let mut raf_checkpoints = vec![Fr::from(0u64); RAF_CHECKPOINTS];
        if CANONICAL_INSTRUCTION_ADDRESS {
            raf_checkpoints[3] = Fr::from(1u64);
        }
        let raf_checkpoints = context.upload(&raf_checkpoints)?;

        let mut phase = Self {
            shards,
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

    #[tracing::instrument(skip_all, name = "ap_init_phase")]
    fn init_phase(&mut self, context: &CudaKernelContext, phase: usize) -> Result<(), CudaError> {
        let v_prev = if phase == 0 {
            None
        } else {
            Some(
                self.v_tables
                    .last()
                    .ok_or(CudaError::InvariantViolation {
                        reason: "a later address phase found no previous eq table",
                    })?
                    .to_host()?,
            )
        };

        let Self {
            shards,
            present,
            address_bits,
            ..
        } = self;
        let address_bits = *address_bits;
        let (raf, suffixes) = if let [shard] = shards.as_mut_slice() {
            let device = context_for(shard.ordinal).ok_or(CudaError::InvariantViolation {
                reason: "an address-phase window names an absent device",
            })?;
            if let Some(v_prev) = &v_prev {
                let v_prev = device.upload(v_prev)?;
                condense_u_evals(
                    device,
                    &shard.rows,
                    &mut shard.u_evals,
                    &v_prev,
                    address_bits,
                    phase,
                )?;
            }
            let raf = init_raf_buckets(device, &shard.rows, &shard.u_evals, address_bits, phase)?;
            let suffixes = init_suffix_columns(
                device,
                &shard.rows,
                &shard.u_evals,
                present,
                address_bits,
                phase,
            )?;
            (raf, suffixes)
        } else {
            let parts =
                shard_phase_tables(shards, present, address_bits, phase, v_prev.as_deref())?;
            let totals = sum_phase_tables(parts)?;
            (
                upload_raf(context, &totals.raf)?,
                context.upload(&totals.suffixes)?,
            )
        };
        let suffix_count = suffixes.len() / CHUNK_SIZE;

        let raf_prefix = self.build_raf_prefix_tables(context, phase)?;

        self.tables = PhaseTables {
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

    #[tracing::instrument(skip_all, name = "ap_evaluate_prefixes")]
    fn evaluate_prefixes(
        &self,
        context: &CudaKernelContext,
        half: usize,
    ) -> Result<DeviceFrVec, CudaError> {
        let b_len = self.tables.len.trailing_zeros().checked_sub(1).ok_or(
            CudaError::InvariantViolation {
                reason: "the address round tables are already fully bound",
            },
        )? as usize;
        prefix_mle_round(
            context,
            &self.checkpoints,
            self.phase_challenges
                .last()
                .copied()
                .unwrap_or(Fr::from(0u64)),
            self.rounds_bound % 2 == 1,
            self.rounds_bound,
            b_len,
            half,
        )
    }

    #[tracing::instrument(skip_all, name = "ap_build_raf_prefix")]
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

    #[tracing::instrument(skip_all, name = "ap_round_message")]
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
        if context.ordinal() != self.layout.ordinal {
            return Err(CudaError::InvariantViolation {
                reason: "an address round message ran on a device the term layout does not live on",
            });
        }
        let prefixes = self.evaluate_prefixes(context, half)?;

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
        let _ = builder.arg(prefixes.limbs());
        let _ = builder.arg(&self.layout.prefix_ids);
        let _ = builder.arg(&self.layout.suffix_slots);
        let _ = builder.arg(&self.layout.scales);
        let _ = builder.arg(&self.layout.offsets);
        let _ = builder.arg(&self.layout.counts);
        let _ = builder.arg(self.tables.suffixes.limbs());
        let _ = builder.arg(&self.layout.suffix_bases);
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

    #[tracing::instrument(skip_all, name = "ap_bind")]
    pub fn bind(&mut self, context: &CudaKernelContext, challenge: Fr) -> Result<(), CudaError> {
        self.tables.bind(context, challenge)?;
        self.phase_challenges.push(challenge);
        self.rounds_bound += 1;

        if self.rounds_bound.is_multiple_of(2) {
            let round = self.rounds_bound - 1;
            let pair = self.phase_challenges.len().checked_sub(2).ok_or(
                CudaError::InvariantViolation {
                    reason: "a checkpoint update pair spans a phase boundary",
                },
            )?;
            let r_x = self.phase_challenges[pair];
            let r_y = self.phase_challenges[pair + 1];
            let suffix_len = self.suffix_len(round / CHUNK_LEN)?;
            self.checkpoints =
                update_checkpoints(context, &self.checkpoints, r_x, r_y, round, suffix_len)?;
        }

        if self.phase_challenges.len() == CHUNK_LEN {
            self.v_tables
                .push(context.eq_evals(&self.phase_challenges)?);
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
        let stride = self.len;
        let mut lanes: Vec<&mut DeviceFrVec> = vec![&mut self.suffixes, &mut self.raf_prefix];
        lanes.extend([
            &mut self.raf.shift_half,
            &mut self.raf.shift_full,
            &mut self.raf.left,
            &mut self.raf.right,
            &mut self.raf.identity,
            &mut self.raf.upper_all_ones,
        ]);
        bind_lanes(context, &mut lanes, stride, challenge)?;
        self.len /= 2;
        Ok(())
    }
}

pub(super) fn bind_lanes(
    context: &CudaKernelContext,
    lanes: &mut [&mut DeviceFrVec],
    stride: usize,
    challenge: Fr,
) -> Result<(), CudaError> {
    if stride < 2 {
        for lane in lanes.iter_mut() {
            **lane = context.alloc(0)?;
        }
        return Ok(());
    }
    let half = stride / 2;
    let mut outputs = Vec::with_capacity(lanes.len());
    let mut counts = Vec::with_capacity(lanes.len());
    for lane in lanes.iter() {
        if !lane.len().is_multiple_of(stride) {
            return Err(CudaError::LengthMismatch {
                expected: stride,
                got: lane.len(),
            });
        }
        let count = (lane.len() / stride) * half;
        counts.push(u32::try_from(count).map_err(|_| CudaError::LengthMismatch {
            expected: u32::MAX as usize,
            got: count,
        })?);
        outputs.push(context.alloc(count)?);
    }
    let max_count = counts.iter().copied().max().unwrap_or(0);
    if max_count == 0 {
        for (lane, out) in lanes.iter_mut().zip(outputs) {
            **lane = out;
        }
        return Ok(());
    }

    let inputs = context.device_pointers(&lanes.iter().map(|lane| &**lane).collect::<Vec<_>>())?;
    let targets = context.device_pointers(&outputs.iter().collect::<Vec<_>>())?;
    let device_counts = context.upload_u32_slice(&counts)?;
    let limbs = fr_limbs(challenge);
    let lane_count = u32::try_from(lanes.len()).map_err(|_| CudaError::LengthMismatch {
        expected: u32::MAX as usize,
        got: lanes.len(),
    })?;
    let half_arg = CudaKernelContext::count_of(half)?;
    let stride_arg = CudaKernelContext::count_of(stride)?;
    let mut builder = context.stream().launch_builder(context.ap_bind_lanes());
    let _ = builder.arg(&inputs);
    let _ = builder.arg(&targets);
    let _ = builder.arg(&device_counts);
    let _ = builder.arg(&limbs[0]);
    let _ = builder.arg(&limbs[1]);
    let _ = builder.arg(&limbs[2]);
    let _ = builder.arg(&limbs[3]);
    let _ = builder.arg(&half_arg);
    let _ = builder.arg(&stride_arg);
    let _ = builder.arg(&max_count);
    // SAFETY: `blockIdx.y` indexes `lanes` because the grid's y extent is
    // `lane_count`, so `counts[lane]`, `in_ptrs[lane]` and `out_ptrs[lane]` are
    // all inside their `lanes.len()`-element uploads. A thread returns unless
    // `i < counts[lane]`, and `counts[lane] == (len / stride) * half` for that
    // lane, so `column = i / half < len / stride` and `b = i % half < half`: the
    // reads `in[column * stride + b]` and `in[column * stride + b + half]` are
    // inside that lane's `len` elements, and the write `out[i]` is inside its
    // freshly allocated `counts[lane]`. Every output is a distinct new
    // allocation, so no lane aliases another's input. The challenge travels by
    // value as four limbs, so no buffer backs it.
    let _ = unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (max_count.div_ceil(BLOCK).max(1), lane_count, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    for (lane, out) in lanes.iter_mut().zip(outputs) {
        **lane = out;
    }
    Ok(())
}

struct ShardPhaseTables {
    raf: Vec<Fr>,
    suffixes: Vec<Fr>,
}

fn shard_phase_tables(
    shards: &mut [AddressShard],
    present: &[LookupTableKind<RISCV_XLEN>],
    address_bits: usize,
    phase: usize,
    v_prev: Option<&[Fr]>,
) -> Result<Vec<ShardPhaseTables>, CudaError> {
    let absent = || CudaError::InvariantViolation {
        reason: "an address-phase window names an absent device",
    };
    let tasks: Vec<DeviceTask<'_, ShardPhaseTables, CudaError>> = shards
        .iter_mut()
        .map(|shard| {
            let task: DeviceTask<'_, ShardPhaseTables, CudaError> = Box::new(move || {
                let device = context_for(shard.ordinal).ok_or_else(absent)?;
                if let Some(v_prev) = v_prev {
                    let v_prev = device.upload(v_prev)?;
                    condense_u_evals(
                        device,
                        &shard.rows,
                        &mut shard.u_evals,
                        &v_prev,
                        address_bits,
                        phase,
                    )?;
                }
                let raf =
                    init_raf_columns(device, &shard.rows, &shard.u_evals, address_bits, phase)?;
                let suffixes = init_suffix_columns(
                    device,
                    &shard.rows,
                    &shard.u_evals,
                    present,
                    address_bits,
                    phase,
                )?;
                Ok(ShardPhaseTables {
                    raf: raf.to_host()?,
                    suffixes: suffixes.to_host()?,
                })
            });
            task
        })
        .collect();
    fan_out(tasks)
}

fn sum_phase_tables(parts: Vec<ShardPhaseTables>) -> Result<ShardPhaseTables, CudaError> {
    let mut parts = parts.into_iter();
    let mut total = parts.next().ok_or(CudaError::InvariantViolation {
        reason: "the address-phase reduce produced no window",
    })?;
    for part in parts {
        for (column, addend) in [
            (&mut total.suffixes, &part.suffixes),
            (&mut total.raf, &part.raf),
        ] {
            if column.len() != addend.len() {
                return Err(CudaError::LengthMismatch {
                    expected: column.len(),
                    got: addend.len(),
                });
            }
            for (slot, value) in column.iter_mut().zip(addend) {
                *slot += *value;
            }
        }
    }
    Ok(total)
}

fn upload_raf(context: &CudaKernelContext, columns: &[Fr]) -> Result<RafBuckets, CudaError> {
    if columns.len() != RAF_LANES * CHUNK_SIZE {
        return Err(CudaError::LengthMismatch {
            expected: RAF_LANES * CHUNK_SIZE,
            got: columns.len(),
        });
    }
    let lane = |index: usize| {
        columns
            .get(index * CHUNK_SIZE..(index + 1) * CHUNK_SIZE)
            .ok_or(CudaError::LengthMismatch {
                expected: RAF_LANES * CHUNK_SIZE,
                got: columns.len(),
            })
            .and_then(|values| context.upload(values))
    };
    Ok(RafBuckets {
        shift_half: lane(0)?,
        shift_full: lane(1)?,
        left: lane(2)?,
        right: lane(3)?,
        identity: lane(4)?,
        upper_all_ones: lane(5)?,
    })
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::Fr;

    use super::bind_lanes;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::device::DeviceFrVec;
    use crate::cuda::common::testing::fr;

    fn host_bind(values: &[Fr], stride: usize, challenge: Fr) -> Vec<Fr> {
        let half = stride / 2;
        let mut out = Vec::with_capacity(values.len() / 2);
        for column in values.chunks_exact(stride) {
            for b in 0..half {
                let (lo, hi) = (column[b], column[b + half]);
                out.push(lo + challenge * (hi - lo));
            }
        }
        out
    }

    #[test]
    fn bind_lanes_matches_the_host_bind() {
        let Some(context) = shared_context() else {
            return;
        };
        const STRIDE: usize = 256;
        const ROUNDS: usize = 3;
        let widths = [37usize, 1, 1, 1, 1, 1, 1, 1];
        let mut host: Vec<Vec<Fr>> = widths
            .iter()
            .enumerate()
            .map(|(lane, &columns)| {
                (0..columns * STRIDE)
                    .map(|i| fr((lane as u64 + 1) * 7_001 + i as u64 + 1))
                    .collect()
            })
            .collect();
        let mut device: Vec<DeviceFrVec> = host
            .iter()
            .map(|values| context.upload(values).expect("upload a lane"))
            .collect();

        let mut stride = STRIDE;
        for round in 0..ROUNDS {
            let challenge = fr(0x00C0_FFEE + round as u64 * 91);
            {
                let mut lanes: Vec<&mut DeviceFrVec> = device.iter_mut().collect();
                bind_lanes(context, &mut lanes, stride, challenge).expect("batched bind");
            }
            for values in &mut host {
                *values = host_bind(values, stride, challenge);
            }
            stride /= 2;

            for (lane, (got, want)) in device.iter().zip(&host).enumerate() {
                assert_eq!(
                    got.to_host().expect("download a bound lane"),
                    *want,
                    "round {round} lane {lane}: the batched bind diverged from the host bind",
                );
            }
        }
        assert!(
            host.iter().all(|values| !values.is_empty())
                && host[0].iter().any(|value| *value != host[0][0]),
            "the fixture collapsed, so the comparison proves nothing",
        );
    }
}
