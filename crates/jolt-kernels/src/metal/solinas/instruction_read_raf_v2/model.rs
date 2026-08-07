//! Checked suffix mapping and analytical slice-A model.

use std::mem::size_of;

use jolt_claims::protocols::jolt::geometry::instruction::CANONICAL_INSTRUCTION_ADDRESS;
use jolt_lookup_tables::tables::suffixes::Suffixes;
use jolt_lookup_tables::{LookupBits, LookupTableKind, XLEN as RISCV_XLEN};
use thiserror::Error;

use super::carrier::{
    segment_selectors, CarrierError, InstructionFactsCarrier, PlaneReceipt, ProducerIdentity,
    ADDRESS_BITS, ADDRESS_PHASES, GROUPED_SEGMENTS, LOOKUP_TABLES, PHASE_BINS, PHASE_BITS,
};

pub const RAF_LANES: usize = 3;
pub const EXPLICIT_SUFFIX_LANES: usize = 3;
pub const FUSED_LANES: usize = RAF_LANES + EXPLICIT_SUFFIX_LANES;
pub const MAX_DECLARED_SUFFIXES: usize = 4;
pub const TOTAL_DECLARED_SUFFIXES: usize = 88;
pub const DEFERRED_WORDS_PER_FIELD: u64 = 5;
pub const MAX_ROWS_PER_JOB: usize = 1 << 16;
pub const SIMD_WIDTH: usize = 32;
pub const PRIMARY_THREADS_PER_THREADGROUP: usize = 1024;
pub const DYNAMIC_THREADGROUP_BYTES: usize =
    FUSED_LANES * PHASE_BINS * DEFERRED_WORDS_PER_FIELD as usize * size_of::<u32>();

pub const LOG_26_CPU_MEMBER_NS: u64 = 3_775_559_408;
pub const LOG_26_FIVE_X_CAP_NS: u64 = 755_111_882;
pub const LOG_26_SEVEN_X_TARGET_NS: u64 = 539_365_630;
pub const LOG_26_GROUPED_ADDRESS_WALL_CAP_NS: u64 = 260_000_000;
pub const RETAINED_COPY_BYTES_PER_SECOND: f64 = 451_701_710_520.0;
pub const RETAINED_PRODUCTS_PER_SECOND: f64 = 16.42e9;

const FIELD_BYTES: u64 = 16;
const LOOKUP_BYTES: u64 = 16;
const INDEX_BYTES: u64 = 4;
const JOB_BYTES: u64 = 16;
const _: () = assert!(DYNAMIC_THREADGROUP_BYTES == 30_720);

/// Mapping from each declared table suffix to one of six segment-local lanes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SuffixLanePlan {
    output_offsets: [u16; LOOKUP_TABLES + 1],
    suffix_counts: [u8; LOOKUP_TABLES],
    explicit_counts: [u8; LOOKUP_TABLES],
    explicit_kinds: [[u8; EXPLICIT_SUFFIX_LANES]; LOOKUP_TABLES],
    declared_to_lane: [[u8; MAX_DECLARED_SUFFIXES]; LOOKUP_TABLES],
}

impl SuffixLanePlan {
    pub fn production() -> Result<Self, ModelError> {
        let mut output_offsets = [0u16; LOOKUP_TABLES + 1];
        let mut suffix_counts = [0u8; LOOKUP_TABLES];
        let mut explicit_counts = [0u8; LOOKUP_TABLES];
        let mut explicit_kinds = [[0u8; EXPLICIT_SUFFIX_LANES]; LOOKUP_TABLES];
        let mut declared_to_lane = [[u8::MAX; MAX_DECLARED_SUFFIXES]; LOOKUP_TABLES];
        let mut total = 0usize;

        for table in LookupTableKind::<RISCV_XLEN>::iter() {
            let table_index = table.index();
            if table_index >= LOOKUP_TABLES {
                return Err(ModelError::InvalidTable(table_index));
            }
            let suffixes = table.suffixes();
            let assignment = assign_suffix_lanes(table_index, suffixes)?;
            suffix_counts[table_index] = suffixes.len() as u8;
            explicit_counts[table_index] = assignment.explicit_count;
            explicit_kinds[table_index] = assignment.explicit_kinds;
            declared_to_lane[table_index] = assignment.declared_to_lane;
            total = total
                .checked_add(suffixes.len())
                .ok_or(ModelError::Overflow)?;
            output_offsets[table_index + 1] =
                u16::try_from(total).map_err(|_| ModelError::Overflow)?;
        }
        if total != TOTAL_DECLARED_SUFFIXES {
            return Err(ModelError::SuffixCount {
                expected: TOTAL_DECLARED_SUFFIXES,
                got: total,
            });
        }
        Ok(Self {
            output_offsets,
            suffix_counts,
            explicit_counts,
            explicit_kinds,
            declared_to_lane,
        })
    }

    pub const fn output_offsets(&self) -> &[u16; LOOKUP_TABLES + 1] {
        &self.output_offsets
    }

    pub const fn suffix_counts(&self) -> &[u8; LOOKUP_TABLES] {
        &self.suffix_counts
    }

    pub const fn explicit_counts(&self) -> &[u8; LOOKUP_TABLES] {
        &self.explicit_counts
    }

    pub const fn explicit_kinds(&self) -> &[[u8; EXPLICIT_SUFFIX_LANES]; LOOKUP_TABLES] {
        &self.explicit_kinds
    }

    pub const fn declared_to_lane(&self) -> &[[u8; MAX_DECLARED_SUFFIXES]; LOOKUP_TABLES] {
        &self.declared_to_lane
    }

    pub fn output_range(&self, table: usize) -> Result<std::ops::Range<usize>, ModelError> {
        if table >= LOOKUP_TABLES {
            return Err(ModelError::InvalidTable(table));
        }
        Ok(self.output_offsets[table] as usize..self.output_offsets[table + 1] as usize)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TableLaneAssignment {
    explicit_count: u8,
    explicit_kinds: [u8; EXPLICIT_SUFFIX_LANES],
    declared_to_lane: [u8; MAX_DECLARED_SUFFIXES],
}

fn assign_suffix_lanes(
    table: usize,
    suffixes: &[Suffixes],
) -> Result<TableLaneAssignment, ModelError> {
    if suffixes.len() > MAX_DECLARED_SUFFIXES {
        return Err(ModelError::TooManySuffixes {
            table,
            count: suffixes.len(),
        });
    }
    let mut explicit_count = 0usize;
    let mut one_seen = false;
    let mut explicit_kinds = [0u8; EXPLICIT_SUFFIX_LANES];
    let mut declared_to_lane = [u8::MAX; MAX_DECLARED_SUFFIXES];
    for (slot, suffix) in suffixes.iter().enumerate() {
        let lane = if *suffix == Suffixes::One {
            if one_seen {
                return Err(ModelError::DuplicateOneSuffix { table });
            }
            one_seen = true;
            0
        } else {
            if explicit_count == EXPLICIT_SUFFIX_LANES {
                return Err(ModelError::TooManyExplicitSuffixes {
                    table,
                    count: explicit_count + 1,
                });
            }
            explicit_kinds[explicit_count] = *suffix as u8;
            let lane = RAF_LANES + explicit_count;
            explicit_count += 1;
            lane
        };
        declared_to_lane[slot] = lane as u8;
    }
    Ok(TableLaneAssignment {
        explicit_count: explicit_count as u8,
        explicit_kinds,
        declared_to_lane,
    })
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PhaseWork {
    pub rows_scanned: u64,
    pub equality_products: u64,
    pub condensation_products: u64,
    pub raf_scalar_products: u64,
    pub suffix_scalar_products: u64,
    pub accumulated_terms: u64,
}

impl PhaseWork {
    pub fn useful_products(self) -> Result<u64, ModelError> {
        self.equality_products
            .checked_add(self.condensation_products)
            .and_then(|value| value.checked_add(self.raf_scalar_products))
            .and_then(|value| value.checked_add(self.suffix_scalar_products))
            .ok_or(ModelError::Overflow)
    }

    fn exact(facts: InstructionFactsCarrier<'_>, phase: usize) -> Result<Self, ModelError> {
        if phase >= ADDRESS_PHASES {
            return Err(ModelError::InvalidPhase(phase));
        }
        let suffix_len = ADDRESS_BITS - (phase + 1) * PHASE_BITS;
        let suffix_mask = if suffix_len == 0 {
            0
        } else {
            (1u128 << suffix_len) - 1
        };
        let upper_suffix_bits = suffix_len.saturating_sub(RISCV_XLEN);
        let tables = LookupTableKind::<RISCV_XLEN>::iter().collect::<Vec<_>>();
        let rows = facts.rows() as u64;
        let mut work = Self {
            rows_scanned: rows,
            equality_products: if phase == 0 { rows } else { 0 },
            condensation_products: if phase == 0 { 0 } else { rows },
            ..Self::default()
        };

        for cycle in 0..facts.rows() {
            let fact = facts.cycle_fact(cycle)?;
            let suffix_bits = fact.lookup() & suffix_mask;
            if fact.raf_flag() {
                work.accumulated_terms += 1;
                if suffix_bits != 0 {
                    work.raf_scalar_products += 1;
                    work.accumulated_terms += 1;
                }
                if CANONICAL_INSTRUCTION_ADDRESS
                    && (upper_suffix_bits == 0
                        || suffix_bits >> (suffix_len - upper_suffix_bits)
                            == (1u128 << upper_suffix_bits) - 1)
                {
                    work.accumulated_terms += 1;
                }
            } else {
                work.accumulated_terms += 1;
                let (left, right) = LookupBits::new(suffix_bits, suffix_len).uninterleave();
                for scalar in [u64::from(left), u64::from(right)] {
                    if scalar != 0 {
                        work.raf_scalar_products += 1;
                        work.accumulated_terms += 1;
                    }
                }
            }

            let Some(table) = fact.table_index() else {
                continue;
            };
            for suffix in tables[table].suffixes() {
                let scalar = suffix.suffix_mle(LookupBits::new(suffix_bits, suffix_len));
                if scalar == 0 {
                    continue;
                }
                if scalar != 1 {
                    work.suffix_scalar_products += 1;
                }
                if *suffix != Suffixes::One {
                    work.accumulated_terms += 1;
                }
            }
        }
        Ok(work)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhaseCensus {
    jobs: u64,
    suffix_finalize_job_lanes: u64,
    work: PhaseWork,
}

impl PhaseCensus {
    pub const fn jobs(self) -> u64 {
        self.jobs
    }

    pub const fn suffix_finalize_job_lanes(self) -> u64 {
        self.suffix_finalize_job_lanes
    }

    pub const fn work(self) -> PhaseWork {
        self.work
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GroupedAddressCensus {
    producer: ProducerIdentity,
    claims_receipt: PlaneReceipt,
    topology_allocation_identity: usize,
    rows: u64,
    selected_rows: u64,
    raf_rows: u64,
    rows_per_job: u64,
    flag_outer_fields: u64,
    flag_equality_fields: u64,
    flag_accumulated_terms: u64,
    segment_rows: [u64; GROUPED_SEGMENTS],
    phases: [PhaseCensus; ADDRESS_PHASES],
}

impl GroupedAddressCensus {
    pub fn from_carrier(
        facts: InstructionFactsCarrier<'_>,
        rows_per_job: usize,
    ) -> Result<Self, ModelError> {
        if rows_per_job == 0 || rows_per_job > MAX_ROWS_PER_JOB || !rows_per_job.is_power_of_two() {
            return Err(ModelError::InvalidRowsPerJob(rows_per_job));
        }
        let suffixes = SuffixLanePlan::production()?;
        let topology = facts.topology();
        let rows = facts.rows() as u64;
        let mut segment_rows = [0u64; GROUPED_SEGMENTS];
        let mut jobs = 0u64;
        let mut suffix_finalize_job_lanes = 0u64;
        let mut selected_rows = 0u64;
        let mut raf_rows = 0u64;
        for (segment, segment_rows_slot) in segment_rows.iter_mut().enumerate() {
            let count = topology
                .segment_len(segment)
                .map_err(|_| ModelError::InvalidSegment(segment))? as u64;
            *segment_rows_slot = count;
            let segment_jobs = count.div_ceil(rows_per_job as u64);
            jobs = jobs.checked_add(segment_jobs).ok_or(ModelError::Overflow)?;
            let Some((table, _raf)) = segment_selectors(segment) else {
                return Err(ModelError::InvalidSegment(segment));
            };
            if segment & 1 != 0 {
                raf_rows = raf_rows.checked_add(count).ok_or(ModelError::Overflow)?;
            }
            if let Some(table) = table {
                selected_rows = selected_rows
                    .checked_add(count)
                    .ok_or(ModelError::Overflow)?;
                suffix_finalize_job_lanes = suffix_finalize_job_lanes
                    .checked_add(segment_jobs * u64::from(suffixes.suffix_counts[table]))
                    .ok_or(ModelError::Overflow)?;
            }
        }
        if jobs == 0 {
            return Err(ModelError::EmptyJobPlan);
        }
        let mut work = [PhaseWork::default(); ADDRESS_PHASES];
        for (phase, phase_work) in work.iter_mut().enumerate() {
            *phase_work = PhaseWork::exact(facts, phase)?;
        }
        let phases = std::array::from_fn(|phase| PhaseCensus {
            jobs,
            suffix_finalize_job_lanes,
            work: work[phase],
        });
        let log_t = facts.rows().ilog2() as usize;
        let out_bits = log_t / 2;
        let in_bits = log_t - out_bits;
        let flag_outer_fields = 1u64 << out_bits;
        let flag_equality_fields = (1u64 << out_bits) + (1u64 << in_bits);
        let flag_accumulated_terms = selected_rows
            .checked_add(raf_rows)
            .ok_or(ModelError::Overflow)?;
        Ok(Self {
            producer: facts.producer(),
            claims_receipt: facts.claims_receipt(),
            topology_allocation_identity: topology.allocation_identity(),
            rows,
            selected_rows,
            raf_rows,
            rows_per_job: rows_per_job as u64,
            flag_outer_fields,
            flag_equality_fields,
            flag_accumulated_terms,
            segment_rows,
            phases,
        })
    }

    pub const fn jobs_per_phase(&self) -> u64 {
        self.phases[0].jobs
    }

    pub const fn producer(&self) -> ProducerIdentity {
        self.producer
    }

    pub const fn claims_receipt(&self) -> PlaneReceipt {
        self.claims_receipt
    }

    pub const fn topology_allocation_identity(&self) -> usize {
        self.topology_allocation_identity
    }

    pub const fn rows(&self) -> u64 {
        self.rows
    }

    pub const fn selected_rows(&self) -> u64 {
        self.selected_rows
    }

    pub const fn raf_rows(&self) -> u64 {
        self.raf_rows
    }

    pub const fn rows_per_job(&self) -> u64 {
        self.rows_per_job
    }

    pub const fn segment_rows(&self) -> &[u64; GROUPED_SEGMENTS] {
        &self.segment_rows
    }

    pub const fn phases(&self) -> &[PhaseCensus; ADDRESS_PHASES] {
        &self.phases
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupedAddressModel {
    address_useful_products: u64,
    flag_useful_products: u64,
    deferred_atomic_word_adds: u64,
    /// Streaming row traffic; small repeatedly indexed tables are excluded.
    large_streaming_bytes: u64,
    cache_requested_bytes: u64,
    partial_transfer_bytes: u64,
    output_transfer_bytes: u64,
    /// Bytes used for the copy roof in the v2 packet.
    roof_issued_bytes: u64,
    logical_requested_bytes: u64,
    compulsory_bytes: u64,
    peak_partial_bytes: u64,
    command_buffers: u64,
    waits: u64,
}

impl GroupedAddressModel {
    pub fn new(census: &GroupedAddressCensus) -> Result<Self, ModelError> {
        let address_useful_products = census.phases.iter().try_fold(0u64, |sum, phase| {
            sum.checked_add(phase.work.useful_products()?)
                .ok_or(ModelError::Overflow)
        })?;
        let flag_useful_products = (LOOKUP_TABLES as u64 + 1)
            .checked_mul(census.flag_outer_fields)
            .ok_or(ModelError::Overflow)?;
        let address_terms = census.phases.iter().try_fold(0u64, |sum, phase| {
            sum.checked_add(phase.work.accumulated_terms)
                .ok_or(ModelError::Overflow)
        })?;
        let deferred_atomic_word_adds = address_terms
            .checked_add(census.flag_accumulated_terms)
            .and_then(|terms| terms.checked_mul(DEFERRED_WORDS_PER_FIELD))
            .ok_or(ModelError::Overflow)?;

        // Phase zero: grouped index + lookup + weight write. Later phases:
        // lookup + weight read + weight write.
        let large_streaming_bytes = (INDEX_BYTES + LOOKUP_BYTES + FIELD_BYTES)
            .checked_mul(census.rows)
            .and_then(|first| {
                (ADDRESS_PHASES as u64 - 1)
                    .checked_mul((LOOKUP_BYTES + 2 * FIELD_BYTES) * census.rows)
                    .and_then(|later| first.checked_add(later))
            })
            .ok_or(ModelError::Overflow)?;
        // Phase zero requests E_in and E_out. Later phases request one 4 KiB
        // phase table entry for every row.
        let cache_requested_bytes = (2 * FIELD_BYTES)
            .checked_mul(census.rows)
            .and_then(|first| {
                (ADDRESS_PHASES as u64 - 1)
                    .checked_mul(FIELD_BYTES * census.rows)
                    .and_then(|later| first.checked_add(later))
            })
            .ok_or(ModelError::Overflow)?;

        let partial_lanes = census.phases.iter().try_fold(0u64, |sum, phase| {
            // Six lane writes, three RAF finalizer reads, then declared suffix reads.
            let lanes = 9u64
                .checked_mul(phase.jobs)
                .and_then(|value| value.checked_add(phase.suffix_finalize_job_lanes))
                .ok_or(ModelError::Overflow)?;
            sum.checked_add(lanes).ok_or(ModelError::Overflow)
        })?;
        let partial_transfer_bytes = partial_lanes
            .checked_mul(PHASE_BINS as u64 * FIELD_BYTES)
            .ok_or(ModelError::Overflow)?;
        let output_fields = (TOTAL_DECLARED_SUFFIXES as u64 + 6) * PHASE_BINS as u64;
        let output_transfer_bytes = 2 * ADDRESS_PHASES as u64 * output_fields * FIELD_BYTES;
        let roof_issued_bytes = large_streaming_bytes
            .checked_add(partial_transfer_bytes)
            .and_then(|value| value.checked_add(output_transfer_bytes))
            .ok_or(ModelError::Overflow)?;
        let logical_requested_bytes = roof_issued_bytes
            .checked_add(cache_requested_bytes)
            .ok_or(ModelError::Overflow)?;
        let peak_partial_bytes = census
            .phases
            .iter()
            .map(|phase| phase.jobs * FUSED_LANES as u64 * PHASE_BINS as u64 * FIELD_BYTES)
            .max()
            .ok_or(ModelError::EmptyJobPlan)?;
        let compact_planes = (LOOKUP_BYTES + 1 + INDEX_BYTES + FIELD_BYTES) * census.rows;
        let equality_tables = census.flag_equality_fields * FIELD_BYTES;
        let phase_tables = (ADDRESS_PHASES as u64 - 1) * PHASE_BINS as u64 * FIELD_BYTES;
        let metadata = census.jobs_per_phase() * JOB_BYTES
            + (GROUPED_SEGMENTS as u64 + 1) * INDEX_BYTES
            + 40 * 4
            + 40 * 3
            + 40;
        let compulsory_bytes = compact_planes
            .checked_add(equality_tables)
            .and_then(|value| value.checked_add(phase_tables))
            .and_then(|value| value.checked_add(metadata))
            .and_then(|value| value.checked_add(peak_partial_bytes))
            .and_then(|value| value.checked_add(output_fields * FIELD_BYTES))
            .ok_or(ModelError::Overflow)?;
        Ok(Self {
            address_useful_products,
            flag_useful_products,
            deferred_atomic_word_adds,
            large_streaming_bytes,
            cache_requested_bytes,
            partial_transfer_bytes,
            output_transfer_bytes,
            roof_issued_bytes,
            logical_requested_bytes,
            compulsory_bytes,
            peak_partial_bytes,
            command_buffers: ADDRESS_PHASES as u64,
            waits: ADDRESS_PHASES as u64,
        })
    }

    pub const fn address_useful_products(self) -> u64 {
        self.address_useful_products
    }

    pub const fn flag_useful_products(self) -> u64 {
        self.flag_useful_products
    }

    pub const fn deferred_atomic_word_adds(self) -> u64 {
        self.deferred_atomic_word_adds
    }

    pub const fn large_streaming_bytes(self) -> u64 {
        self.large_streaming_bytes
    }

    pub const fn cache_requested_bytes(self) -> u64 {
        self.cache_requested_bytes
    }

    pub const fn partial_transfer_bytes(self) -> u64 {
        self.partial_transfer_bytes
    }

    pub const fn output_transfer_bytes(self) -> u64 {
        self.output_transfer_bytes
    }

    pub const fn roof_issued_bytes(self) -> u64 {
        self.roof_issued_bytes
    }

    pub const fn logical_requested_bytes(self) -> u64 {
        self.logical_requested_bytes
    }

    pub const fn compulsory_bytes(self) -> u64 {
        self.compulsory_bytes
    }

    pub const fn peak_partial_bytes(self) -> u64 {
        self.peak_partial_bytes
    }

    pub const fn command_buffers(self) -> u64 {
        self.command_buffers
    }

    pub const fn waits(self) -> u64 {
        self.waits
    }

    pub fn roof(self, controls: RoofControls) -> Result<RoofEstimate, ModelError> {
        controls.validate()?;
        let copy_floor_ns = rate_floor_ns(self.roof_issued_bytes, controls.copy_bytes_per_second);
        let products = self
            .address_useful_products
            .checked_add(self.flag_useful_products)
            .ok_or(ModelError::Overflow)?;
        let product_floor_ns = rate_floor_ns(products, controls.products_per_second);
        let atomic_floor_ns = rate_floor_ns(
            self.deferred_atomic_word_adds,
            controls.atomic_word_adds_per_second,
        );
        let limiting_floor_ns = copy_floor_ns.max(product_floor_ns).max(atomic_floor_ns);
        let command_ns = self
            .command_buffers
            .checked_mul(controls.command_floor_ns)
            .ok_or(ModelError::Overflow)?;
        let eighty_percent_cap_ns = limiting_floor_ns
            .checked_mul(5)
            .ok_or(ModelError::Overflow)?
            .div_ceil(4)
            .checked_add(command_ns)
            .ok_or(ModelError::Overflow)?;
        Ok(RoofEstimate {
            copy_floor_ns,
            product_floor_ns,
            atomic_floor_ns,
            limiting_floor_ns,
            eighty_percent_cap_ns,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoofControls {
    pub copy_bytes_per_second: f64,
    pub products_per_second: f64,
    pub atomic_word_adds_per_second: f64,
    pub command_floor_ns: u64,
}

impl RoofControls {
    fn validate(self) -> Result<(), ModelError> {
        for (name, value) in [
            ("copy bytes", self.copy_bytes_per_second),
            ("field products", self.products_per_second),
            ("atomic word adds", self.atomic_word_adds_per_second),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(ModelError::InvalidRate { name });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoofEstimate {
    pub copy_floor_ns: u64,
    pub product_floor_ns: u64,
    pub atomic_floor_ns: u64,
    pub limiting_floor_ns: u64,
    pub eighty_percent_cap_ns: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AddressLaunchGeometry {
    pub threads_per_threadgroup: u64,
    pub simdgroups_per_threadgroup: u64,
    pub dynamic_threadgroup_bytes: u64,
    pub threadgroups: u64,
    pub scheduled_waves: u64,
}

impl AddressLaunchGeometry {
    pub fn primary(census: &GroupedAddressCensus, gpu_cores: u64) -> Result<Self, ModelError> {
        if gpu_cores == 0 {
            return Err(ModelError::InvalidGpuCores);
        }
        let threadgroups = census.jobs_per_phase();
        Ok(Self {
            threads_per_threadgroup: PRIMARY_THREADS_PER_THREADGROUP as u64,
            simdgroups_per_threadgroup: (PRIMARY_THREADS_PER_THREADGROUP / SIMD_WIDTH) as u64,
            dynamic_threadgroup_bytes: DYNAMIC_THREADGROUP_BYTES as u64,
            threadgroups,
            scheduled_waves: threadgroups.div_ceil(gpu_cores),
        })
    }

    /// Requires compiler register evidence; launch width alone is not occupancy.
    pub fn resident_groups(self, evidence: OccupancyEvidence) -> Result<u64, ModelError> {
        evidence.validate()?;
        let registers_per_group = evidence
            .register_words_per_thread
            .checked_mul(self.threads_per_threadgroup)
            .ok_or(ModelError::Overflow)?;
        let register_groups = evidence.register_words_per_core / registers_per_group;
        Ok(
            (evidence.resident_threads_per_core / self.threads_per_threadgroup)
                .min(evidence.threadgroup_memory_bytes_per_core / self.dynamic_threadgroup_bytes)
                .min(register_groups)
                .min(evidence.architectural_groups_per_core),
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OccupancyEvidence {
    pub resident_threads_per_core: u64,
    pub threadgroup_memory_bytes_per_core: u64,
    pub register_words_per_core: u64,
    pub register_words_per_thread: u64,
    pub architectural_groups_per_core: u64,
}

impl OccupancyEvidence {
    fn validate(self) -> Result<(), ModelError> {
        if [
            self.resident_threads_per_core,
            self.threadgroup_memory_bytes_per_core,
            self.register_words_per_core,
            self.register_words_per_thread,
            self.architectural_groups_per_core,
        ]
        .contains(&0)
        {
            return Err(ModelError::IncompleteOccupancyEvidence);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SliceAMeasurement {
    pub transcript_layout_and_address_ns: u64,
    pub projected_complete_member_ns: u64,
}

impl SliceAMeasurement {
    pub const fn clears_kill_gates(self) -> bool {
        self.transcript_layout_and_address_ns <= LOG_26_GROUPED_ADDRESS_WALL_CAP_NS
            && self.projected_complete_member_ns <= LOG_26_FIVE_X_CAP_NS
    }
}

fn rate_floor_ns(work: u64, per_second: f64) -> u64 {
    (work as f64 / per_second * 1e9).ceil() as u64
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum ModelError {
    #[error(transparent)]
    Carrier(#[from] CarrierError),
    #[error("InstructionReadRaf table {0} is outside the 40-table specialization")]
    InvalidTable(usize),
    #[error("InstructionReadRaf segment {0} is outside the 82-segment topology")]
    InvalidSegment(usize),
    #[error("InstructionReadRaf table {table} declares {count} suffixes; at most four fit")]
    TooManySuffixes { table: usize, count: usize },
    #[error("InstructionReadRaf table {table} declares Suffixes::One more than once")]
    DuplicateOneSuffix { table: usize },
    #[error(
        "InstructionReadRaf table {table} needs {count} explicit suffix lanes; at most three fit"
    )]
    TooManyExplicitSuffixes { table: usize, count: usize },
    #[error("InstructionReadRaf suffix registry has {got} outputs, expected {expected}")]
    SuffixCount { expected: usize, got: usize },
    #[error("InstructionReadRaf rows per job must be a power of two in 1..=65536, got {0}")]
    InvalidRowsPerJob(usize),
    #[error("InstructionReadRaf grouped job plan is empty")]
    EmptyJobPlan,
    #[error("InstructionReadRaf address phase {0} is outside 0..16")]
    InvalidPhase(usize),
    #[error("InstructionReadRaf {name} control rate is missing or invalid")]
    InvalidRate { name: &'static str },
    #[error("InstructionReadRaf GPU core count is zero")]
    InvalidGpuCores,
    #[error("InstructionReadRaf occupancy needs nonzero device and compiled-register evidence")]
    IncompleteOccupancyEvidence,
    #[error("InstructionReadRaf model arithmetic overflowed")]
    Overflow,
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests use fixed valid receipts")]
mod tests {
    use super::*;
    use crate::metal::solinas::instruction_read_raf_v2::carrier::{
        pack_claim, CycleOrderPlane, GroupedAddressTopology, InstructionFactsCarrier, PlaneReceipt,
        ProducerIdentity,
    };

    #[test]
    fn suffix_plan_handles_one_in_any_declared_slot() {
        let assignment = assign_suffix_lanes(
            0,
            &[
                Suffixes::Xor,
                Suffixes::One,
                Suffixes::Eq,
                Suffixes::LessThan,
            ],
        )
        .unwrap();
        assert_eq!(assignment.explicit_count, 3);
        assert_eq!(assignment.declared_to_lane, [3, 0, 4, 5]);
        assert_eq!(
            assignment.explicit_kinds,
            [
                Suffixes::Xor as u8,
                Suffixes::Eq as u8,
                Suffixes::LessThan as u8
            ]
        );
        assert!(matches!(
            assign_suffix_lanes(0, &[Suffixes::One, Suffixes::One]),
            Err(ModelError::DuplicateOneSuffix { table: 0 })
        ));
        assert!(matches!(
            assign_suffix_lanes(
                0,
                &[
                    Suffixes::Xor,
                    Suffixes::Eq,
                    Suffixes::LessThan,
                    Suffixes::And,
                ],
            ),
            Err(ModelError::TooManyExplicitSuffixes { table: 0, count: 4 })
        ));
        assert!(matches!(
            assign_suffix_lanes(
                0,
                &[
                    Suffixes::One,
                    Suffixes::Xor,
                    Suffixes::Eq,
                    Suffixes::LessThan,
                    Suffixes::And,
                ],
            ),
            Err(ModelError::TooManySuffixes { table: 0, count: 5 })
        ));

        let production = SuffixLanePlan::production().unwrap();
        assert_eq!(production.output_offsets()[LOOKUP_TABLES] as usize, 88);
        for table in LookupTableKind::<RISCV_XLEN>::iter() {
            let table_index = table.index();
            let mut explicit = 0usize;
            for (slot, suffix) in table.suffixes().iter().enumerate() {
                let expected = if *suffix == Suffixes::One {
                    0
                } else {
                    let lane = RAF_LANES + explicit;
                    explicit += 1;
                    lane
                };
                assert_eq!(
                    production.declared_to_lane()[table_index][slot] as usize,
                    expected
                );
            }
            assert_eq!(production.explicit_counts()[table_index] as usize, explicit);
            assert_eq!(
                production.output_range(table_index).unwrap().len(),
                table.suffixes().len()
            );
        }
    }

    #[test]
    fn census_derives_work_and_flag_counts_from_receipted_facts() {
        let lookups = [0u128, 0];
        let claims = [
            pack_claim(None, false).unwrap(),
            pack_claim(None, true).unwrap(),
        ];
        let producer = ProducerIdentity::new(5, 0x1000, 6, lookups.len()).unwrap();
        let lookup_plane = CycleOrderPlane::new(
            &lookups,
            PlaneReceipt::new(producer, 0x2000, "lookup plane").unwrap(),
            "lookup plane",
        )
        .unwrap();
        let claim_plane = CycleOrderPlane::new(
            &claims,
            PlaneReceipt::new(producer, 0x3000, "claim plane").unwrap(),
            "claim plane",
        )
        .unwrap();
        let topology = GroupedAddressTopology::stable_from_claims(claim_plane, 0x4000).unwrap();
        let facts =
            InstructionFactsCarrier::attach(5, lookup_plane, claim_plane, &topology).unwrap();
        let census = GroupedAddressCensus::from_carrier(facts, 1).unwrap();

        assert_eq!(census.rows(), 2);
        assert_eq!(census.selected_rows(), 0);
        assert_eq!(census.raf_rows(), 1);
        assert_eq!(census.jobs_per_phase(), 2);
        assert_eq!(census.claims_receipt(), claim_plane.receipt());
        assert_eq!(census.topology_allocation_identity(), 0x4000);
        for (phase, phase_census) in census.phases().iter().copied().enumerate() {
            let work = phase_census.work();
            assert_eq!(work.rows_scanned, 2);
            assert_eq!(work.equality_products, u64::from(phase == 0) * 2);
            assert_eq!(work.condensation_products, u64::from(phase != 0) * 2);
            assert_eq!(work.raf_scalar_products, 0);
            assert_eq!(work.suffix_scalar_products, 0);
            let canonical_term = u64::from(CANONICAL_INSTRUCTION_ADDRESS && phase >= 7);
            assert_eq!(work.accumulated_terms, 2 + canonical_term);
        }

        let model = GroupedAddressModel::new(&census).unwrap();
        let accumulated = census
            .phases()
            .iter()
            .map(|phase| phase.work().accumulated_terms)
            .sum::<u64>()
            + 1;
        assert_eq!(
            model.deferred_atomic_word_adds(),
            accumulated * DEFERRED_WORDS_PER_FIELD
        );
        assert_eq!(model.flag_useful_products(), LOOKUP_TABLES as u64 + 1);
        assert_eq!(model.command_buffers(), ADDRESS_PHASES as u64);
        assert_eq!(model.waits(), ADDRESS_PHASES as u64);
    }

    #[test]
    fn roofs_and_occupancy_fail_closed_on_invalid_controls() {
        let model = GroupedAddressModel {
            address_useful_products: 100,
            flag_useful_products: 10,
            deferred_atomic_word_adds: 20,
            large_streaming_bytes: 1_000,
            cache_requested_bytes: 0,
            partial_transfer_bytes: 0,
            output_transfer_bytes: 0,
            roof_issued_bytes: 1_000,
            logical_requested_bytes: 1_000,
            compulsory_bytes: 1_000,
            peak_partial_bytes: 0,
            command_buffers: 16,
            waits: 16,
        };
        assert!(matches!(
            model.roof(RoofControls {
                copy_bytes_per_second: 0.0,
                products_per_second: 1.0,
                atomic_word_adds_per_second: 1.0,
                command_floor_ns: 0,
            }),
            Err(ModelError::InvalidRate { name: "copy bytes" })
        ));

        let geometry = AddressLaunchGeometry {
            threads_per_threadgroup: u64::MAX,
            simdgroups_per_threadgroup: 1,
            dynamic_threadgroup_bytes: 1,
            threadgroups: 1,
            scheduled_waves: 1,
        };
        assert_eq!(
            geometry.resident_groups(OccupancyEvidence {
                resident_threads_per_core: u64::MAX,
                threadgroup_memory_bytes_per_core: 1,
                register_words_per_core: u64::MAX,
                register_words_per_thread: 2,
                architectural_groups_per_core: 1,
            }),
            Err(ModelError::Overflow)
        );
    }
}
