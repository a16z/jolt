//! Exact target work model and fail-closed experiment gates.

use super::abi::{
    FIELD_BYTES, NATIVE_WORD_BYTES, SIMD_WIDTH, TARGET_ADDRESSES, TARGET_BLOCKS,
    TARGET_BLOCK_ELEMENTS, TARGET_CHALLENGES, TARGET_CHUNKS_PER_BLOCK, TARGET_PARTIALS,
    TARGET_THREADS,
};

pub const SCREENING_EVIDENCE_JSON: &str = include_str!("screening_evidence.json");

pub const FROZEN_CPU_EVIDENCE: &str =
    "crates/jolt-kernels/autoresearch/evidence/ram_output_check_cpu_deferred_log13_observed_821665b4b.json";
pub const FROZEN_CPU_EVIDENCE_SHA256: &str =
    "ebf03a8f4ea5acadca7f3ae8e32e8469d9de14f4f34127ec913b1dfb6dc5bb3f";
pub const FROZEN_CPU_SCREEN_NS: u64 = 276_100;
pub const FIVE_X_SCREEN_CAP_NS: u64 = FROZEN_CPU_SCREEN_NS / 5;
pub const EIGHT_X_SCREEN_CAP_NS: u64 = FROZEN_CPU_SCREEN_NS / 8;

pub const COPY_BYTES_PER_SECOND: u64 = 451_701_710_520;
pub const FULL_PRODUCTS_PER_SECOND: u64 = 45_709_000_000;
pub const HALF_WIDTH_PRODUCTS_PER_SECOND: u64 = 86_592_000_000;

pub const HOST_WEIGHT_CAP_NS: u64 = 6_000;
pub const PARTIAL_DISPATCH_CAP_NS: u64 = 18_000;
pub const HOST_TAIL_CAP_NS: u64 = 8_000;
pub const COMPLETE_EIGHT_X_PURSUIT_NS: u64 = 32_000;
pub const MAX_SERVICE_FLOOR_OVERHEAD_NUMERATOR: u64 = 5;
pub const MAX_SERVICE_FLOOR_OVERHEAD_DENOMINATOR: u64 = 4;

const NANOS_PER_SECOND: u128 = 1_000_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Geometry {
    pub addresses: u32,
    pub block_elements: u32,
    pub blocks: u32,
    pub chunks_per_block: u32,
    pub threads: u32,
    pub zero_rounds: u32,
}

impl Geometry {
    pub const fn target() -> Self {
        Self {
            addresses: TARGET_ADDRESSES,
            block_elements: TARGET_BLOCK_ELEMENTS,
            blocks: TARGET_BLOCKS,
            chunks_per_block: TARGET_CHUNKS_PER_BLOCK,
            threads: TARGET_THREADS,
            zero_rounds: TARGET_CHALLENGES,
        }
    }

    pub fn validate(self) -> Result<Self, ModelError> {
        if self.addresses != TARGET_ADDRESSES
            || self.block_elements != TARGET_BLOCK_ELEMENTS
            || self.blocks != TARGET_BLOCKS
            || self.chunks_per_block != TARGET_CHUNKS_PER_BLOCK
            || self.threads != TARGET_THREADS
            || self.zero_rounds != TARGET_CHALLENGES
            || self.blocks.checked_mul(self.block_elements) != Some(self.addresses)
            || self.chunks_per_block.checked_mul(self.threads) != Some(self.block_elements)
        {
            return Err(ModelError::InvalidGeometry);
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WeightSource {
    IncrementalHostTable,
    DeviceChallenges,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReductionOwner {
    Host,
    Device,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Schedule {
    pub weights: WeightSource,
    pub reduction: ReductionOwner,
}

impl Schedule {
    pub const fn selected() -> Self {
        Self {
            weights: WeightSource::IncrementalHostTable,
            reduction: ReductionOwner::Host,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkPlan {
    pub host_full_products: u128,
    pub device_full_products: u128,
    pub device_half_width_products: u128,
    pub device_reduction_additions: u128,
    pub host_reduction_additions: u128,
    pub threadgroups: u128,
    pub simdgroups: u128,
    pub dispatches: u128,
    pub host_write_bytes: u128,
    pub device_unique_read_bytes: u128,
    pub device_requested_read_bytes: u128,
    pub device_write_bytes: u128,
    pub host_read_bytes: u128,
}

impl WorkPlan {
    pub fn perfect_cache_bytes(self) -> Result<u128, ModelError> {
        checked_sum(&[
            self.host_write_bytes,
            self.device_unique_read_bytes,
            self.device_write_bytes,
            self.host_read_bytes,
        ])
    }

    pub fn shader_requested_bytes(self) -> Result<u128, ModelError> {
        checked_sum(&[
            self.host_write_bytes,
            self.device_requested_read_bytes,
            self.device_write_bytes,
            self.host_read_bytes,
        ])
    }

    pub fn arithmetic_floor_ns(self) -> Result<u64, ModelError> {
        let full = work_floor_ns(self.device_full_products, FULL_PRODUCTS_PER_SECOND)?;
        let half = work_floor_ns(
            self.device_half_width_products,
            HALF_WIDTH_PRODUCTS_PER_SECOND,
        )?;
        full.checked_add(half).ok_or(ModelError::Overflow)
    }

    pub fn perfect_cache_traffic_floor_ns(self) -> Result<u64, ModelError> {
        work_floor_ns(self.perfect_cache_bytes()?, COPY_BYTES_PER_SECOND)
    }

    pub fn requested_traffic_floor_ns(self) -> Result<u64, ModelError> {
        work_floor_ns(self.shader_requested_bytes()?, COPY_BYTES_PER_SECOND)
    }

    pub fn optimistic_device_floor_ns(self) -> Result<u64, ModelError> {
        Ok(self
            .arithmetic_floor_ns()?
            .max(self.perfect_cache_traffic_floor_ns()?))
    }
}

pub fn work_plan(geometry: Geometry, schedule: Schedule) -> Result<WorkPlan, ModelError> {
    let geometry = geometry.validate()?;
    let addresses = u128::from(geometry.addresses);
    let block_elements = u128::from(geometry.block_elements);
    let blocks = u128::from(geometry.blocks);
    let chunks = u128::from(geometry.chunks_per_block);
    let threads = u128::from(geometry.threads);
    let partials = blocks.checked_mul(chunks).ok_or(ModelError::Overflow)?;
    if partials != u128::from(TARGET_PARTIALS)
        || chunks.checked_mul(threads) != Some(block_elements)
    {
        return Err(ModelError::InvalidGeometry);
    }

    let host_weight_table = schedule.weights == WeightSource::IncrementalHostTable;
    let host_reduction = schedule.reduction == ReductionOwner::Host;
    let coefficient_elements = if host_weight_table {
        block_elements
    } else {
        u128::from(geometry.zero_rounds)
    };
    let coefficient_unique_bytes = coefficient_elements
        .checked_mul(u128::from(FIELD_BYTES))
        .ok_or(ModelError::Overflow)?;
    let coefficient_requested_bytes = if host_weight_table {
        addresses
            .checked_mul(u128::from(FIELD_BYTES))
            .ok_or(ModelError::Overflow)?
    } else {
        addresses
            .checked_mul(u128::from(geometry.zero_rounds))
            .and_then(|value| value.checked_mul(u128::from(FIELD_BYTES)))
            .ok_or(ModelError::Overflow)?
    };
    let source_bytes = addresses
        .checked_mul(u128::from(NATIVE_WORD_BYTES))
        .ok_or(ModelError::Overflow)?;
    let partial_bytes = partials
        .checked_mul(u128::from(FIELD_BYTES))
        .ok_or(ModelError::Overflow)?;
    let output_bytes = blocks
        .checked_mul(u128::from(FIELD_BYTES))
        .ok_or(ModelError::Overflow)?;

    let device_full_products = if host_weight_table {
        0
    } else {
        addresses
            .checked_mul(u128::from(geometry.zero_rounds - 1))
            .ok_or(ModelError::Overflow)?
    };
    let reduction_additions = blocks.checked_mul(chunks - 1).ok_or(ModelError::Overflow)?;
    let partial_additions = addresses
        .checked_sub(partials)
        .ok_or(ModelError::Overflow)?;
    let device_reduction_additions = if host_reduction {
        partial_additions
    } else {
        partial_additions
            .checked_add(reduction_additions)
            .ok_or(ModelError::Overflow)?
    };
    let host_reduction_additions = if host_reduction {
        reduction_additions
    } else {
        0
    };

    let reduction_device_read = if host_reduction { 0 } else { partial_bytes };
    let device_unique_read_bytes = checked_sum(&[
        source_bytes,
        coefficient_unique_bytes,
        reduction_device_read,
    ])?;
    let device_requested_read_bytes = checked_sum(&[
        source_bytes,
        coefficient_requested_bytes,
        reduction_device_read,
    ])?;
    let device_write_bytes = if host_reduction {
        partial_bytes
    } else {
        partial_bytes
            .checked_add(output_bytes)
            .ok_or(ModelError::Overflow)?
    };
    let host_read_bytes = if host_reduction {
        partial_bytes
    } else {
        output_bytes
    };

    Ok(WorkPlan {
        host_full_products: if host_weight_table {
            block_elements - 1
        } else {
            0
        },
        device_full_products,
        device_half_width_products: addresses,
        device_reduction_additions,
        host_reduction_additions,
        threadgroups: partials,
        simdgroups: partials
            .checked_mul(threads / u128::from(SIMD_WIDTH))
            .ok_or(ModelError::Overflow)?,
        dispatches: if host_reduction { 1 } else { 2 },
        host_write_bytes: coefficient_unique_bytes,
        device_unique_read_bytes,
        device_requested_read_bytes,
        device_write_bytes,
        host_read_bytes,
    })
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompiledEvidence {
    pub thread_execution_width: Option<u32>,
    pub max_threads_per_threadgroup: Option<u32>,
    pub spills_detected: Option<bool>,
    pub resident_threadgroups_per_core: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TimingEvidence {
    pub parity_passed: bool,
    pub same_parent_command_control: bool,
    pub counter_delimited_auxiliary: bool,
    pub new_command_buffers: u32,
    pub new_waits: u32,
    pub empty_auxiliary_service_ns: Option<u64>,
    pub host_weights_ns: Option<u64>,
    pub partial_dispatch_ns: Option<u64>,
    pub host_tail_ns: Option<u64>,
    pub complete_incremental_ns: Option<u64>,
    pub resident_cpu_ns: Option<u64>,
    pub comparison_noise_ns: Option<u64>,
    pub five_alternating_pairs: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionDecision {
    ParityMissing,
    StandaloneTopologyRejected,
    ParentCommandMismatch,
    CounterBoundaryMissing,
    CompiledLimitsMissing,
    WrongExecutionWidth,
    ThreadgroupWidthUnsupported,
    SpillDetected,
    ResidencyMissing,
    InsufficientResidency,
    TimingMissing,
    HostWeightsTooSlow,
    PartialDispatchTooSlow,
    TailTooSlow,
    ServiceFloorMiss,
    BelowFiveX,
    ResidentCpuWins,
    FiveXOnlyNeedsCeilingReview,
    EightXBudgetMissNeedsReview,
    AlternatingPairsMissing,
    EightXCandidate,
}

pub fn admission_decision(
    work: WorkPlan,
    compiled: CompiledEvidence,
    timing: TimingEvidence,
) -> Result<AdmissionDecision, ModelError> {
    if !timing.parity_passed {
        return Ok(AdmissionDecision::ParityMissing);
    }
    if timing.new_command_buffers != 0 || timing.new_waits != 0 {
        return Ok(AdmissionDecision::StandaloneTopologyRejected);
    }
    if !timing.same_parent_command_control {
        return Ok(AdmissionDecision::ParentCommandMismatch);
    }
    if !timing.counter_delimited_auxiliary {
        return Ok(AdmissionDecision::CounterBoundaryMissing);
    }
    let (Some(execution_width), Some(max_threads), Some(spills), Some(resident)) = (
        compiled.thread_execution_width,
        compiled.max_threads_per_threadgroup,
        compiled.spills_detected,
        compiled.resident_threadgroups_per_core,
    ) else {
        return Ok(AdmissionDecision::CompiledLimitsMissing);
    };
    if execution_width != SIMD_WIDTH {
        return Ok(AdmissionDecision::WrongExecutionWidth);
    }
    if max_threads < TARGET_THREADS {
        return Ok(AdmissionDecision::ThreadgroupWidthUnsupported);
    }
    if spills {
        return Ok(AdmissionDecision::SpillDetected);
    }
    if resident == 0 {
        return Ok(AdmissionDecision::ResidencyMissing);
    }
    if resident < 2 {
        return Ok(AdmissionDecision::InsufficientResidency);
    }
    let (
        Some(service),
        Some(host_weights),
        Some(partial),
        Some(host_tail),
        Some(complete),
        Some(resident_cpu),
        Some(comparison_noise),
    ) = (
        timing.empty_auxiliary_service_ns,
        timing.host_weights_ns,
        timing.partial_dispatch_ns,
        timing.host_tail_ns,
        timing.complete_incremental_ns,
        timing.resident_cpu_ns,
        timing.comparison_noise_ns,
    )
    else {
        return Ok(AdmissionDecision::TimingMissing);
    };
    if host_weights > HOST_WEIGHT_CAP_NS {
        return Ok(AdmissionDecision::HostWeightsTooSlow);
    }
    if partial > PARTIAL_DISPATCH_CAP_NS {
        return Ok(AdmissionDecision::PartialDispatchTooSlow);
    }
    if host_tail > HOST_TAIL_CAP_NS {
        return Ok(AdmissionDecision::TailTooSlow);
    }
    let service_allowance = u128::from(service)
        .checked_mul(u128::from(MAX_SERVICE_FLOOR_OVERHEAD_NUMERATOR))
        .ok_or(ModelError::Overflow)?
        .div_ceil(u128::from(MAX_SERVICE_FLOOR_OVERHEAD_DENOMINATOR));
    let service_gate = service_allowance
        .checked_add(u128::from(work.optimistic_device_floor_ns()?))
        .ok_or(ModelError::Overflow)?;
    if u128::from(partial) > service_gate {
        return Ok(AdmissionDecision::ServiceFloorMiss);
    }
    if complete > FIVE_X_SCREEN_CAP_NS {
        return Ok(AdmissionDecision::BelowFiveX);
    }
    let metal_with_noise = complete
        .checked_add(comparison_noise)
        .ok_or(ModelError::Overflow)?;
    if metal_with_noise >= resident_cpu {
        return Ok(AdmissionDecision::ResidentCpuWins);
    }
    if complete > EIGHT_X_SCREEN_CAP_NS {
        return Ok(AdmissionDecision::FiveXOnlyNeedsCeilingReview);
    }
    if complete > COMPLETE_EIGHT_X_PURSUIT_NS {
        return Ok(AdmissionDecision::EightXBudgetMissNeedsReview);
    }
    if !timing.five_alternating_pairs {
        return Ok(AdmissionDecision::AlternatingPairsMissing);
    }
    Ok(AdmissionDecision::EightXCandidate)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelError {
    InvalidGeometry,
    InvalidRate,
    Overflow,
}

fn checked_sum(values: &[u128]) -> Result<u128, ModelError> {
    values.iter().try_fold(0_u128, |sum, &value| {
        sum.checked_add(value).ok_or(ModelError::Overflow)
    })
}

fn work_floor_ns(work: u128, rate: u64) -> Result<u64, ModelError> {
    if rate == 0 {
        return Err(ModelError::InvalidRate);
    }
    let numerator = work
        .checked_mul(NANOS_PER_SECOND)
        .ok_or(ModelError::Overflow)?;
    let nanos = numerator.div_ceil(u128::from(rate));
    u64::try_from(nanos).map_err(|_| ModelError::Overflow)
}
