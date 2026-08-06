//! Checked geometry, roof model, hybrid rule, and fail-closed promotion gates.

use core::cmp::Ordering;
use core::mem::size_of;

use super::abi::{
    InstructionInputSuccessorDenseMessageParams, InstructionInputSuccessorError,
    InstructionInputSuccessorMaterializeParams, InstructionInputSuccessorRow,
    INSTRUCTION_INPUT_SUCCESSOR_COEFFICIENTS, INSTRUCTION_INPUT_SUCCESSOR_SIMD_WIDTH,
    INSTRUCTION_INPUT_SUCCESSOR_TABLES,
};

pub const TARGET_ROWS: u64 = 1 << 26;
pub const FROZEN_CPU_CUTOFF: u64 = 1 << 16;
pub const FROZEN_TRACE_ADMISSION_LOG: u32 = 25;

pub const FROZEN_CPU_MEDIAN_NS: u64 = 727_212_419;
pub const FROZEN_METAL_MEDIAN_NS: u64 = 142_462_799;
pub const FROZEN_READBACK_MEDIAN_NS: u64 = 1_300_000;
pub const FROZEN_CPU_TAIL_MEDIAN_NS: u64 = 5_522_000;
pub const FROZEN_NON_ROUND_ONE_MEDIAN_NS: u64 = 78_216_632;

pub const RETAINED_COPY_BYTES_PER_SECOND: u64 = 451_701_710_520;
pub const RETAINED_REGISTER_PRODUCTS_PER_SECOND: u64 = 18_100_000_000;
pub const RETAINED_MESSAGE_PRODUCTS_PER_SECOND: u64 = 32_330_000_000;
pub const RETAINED_CONSERVATIVE_PRODUCTS_PER_SECOND: u64 = 16_420_000_000;

pub const PRIMARY_COMPLETE_SERVICE_TARGET_NS: u64 = 122_100_000;
pub const PRIMARY_ROUND_ONE_FALSIFIER_NS: u64 = 45_000_000;

const FP128_BYTES: u128 = 16;
const NANOS_PER_SECOND: u128 = 1_000_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Geometry {
    pub rows: u64,
    pub cpu_cutoff: u64,
}

impl Geometry {
    pub fn new(rows: u64, cpu_cutoff: u64) -> Result<Self, InstructionInputSuccessorError> {
        if rows < 4 || !rows.is_power_of_two() {
            return Err(InstructionInputSuccessorError::InvalidRows);
        }
        if cpu_cutoff == 0 || !cpu_cutoff.is_power_of_two() || cpu_cutoff > rows / 2 {
            return Err(InstructionInputSuccessorError::InvalidCutoff);
        }
        Ok(Self { rows, cpu_cutoff })
    }

    pub fn dense_source_span(self) -> Result<u128, InstructionInputSuccessorError> {
        u128::from(self.rows)
            .checked_sub(
                u128::from(self.cpu_cutoff)
                    .checked_mul(2)
                    .ok_or(InstructionInputSuccessorError::GeometryOverflow)?,
            )
            .ok_or(InstructionInputSuccessorError::GeometryOverflow)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaterializeShape {
    params: InstructionInputSuccessorMaterializeParams,
    grid_threads: usize,
    resident_row_bytes: u64,
    dense_table_bytes: u64,
}

impl MaterializeShape {
    pub(crate) const fn params(self) -> InstructionInputSuccessorMaterializeParams {
        self.params
    }

    pub const fn grid_threads(self) -> usize {
        self.grid_threads
    }

    pub const fn resident_row_bytes(self) -> u64 {
        self.resident_row_bytes
    }

    pub const fn dense_table_bytes(self) -> u64 {
        self.dense_table_bytes
    }
}

pub fn checked_materialize_shape(
    rows: usize,
    max_buffer_length: u64,
) -> Result<MaterializeShape, InstructionInputSuccessorError> {
    if rows < 4 || !rows.is_power_of_two() {
        return Err(InstructionInputSuccessorError::InvalidRows);
    }
    let source_elements =
        u32::try_from(rows).map_err(|_| InstructionInputSuccessorError::ShaderIndexOverflow)?;
    let bound_elements = source_elements / 2;
    let resident_row_bytes = checked_buffer_bytes(
        rows,
        size_of::<InstructionInputSuccessorRow>(),
        max_buffer_length,
    )?;
    let dense_values = rows
        .checked_div(2)
        .and_then(|bound| bound.checked_mul(INSTRUCTION_INPUT_SUCCESSOR_TABLES))
        .ok_or(InstructionInputSuccessorError::GeometryOverflow)?;
    let dense_values_u64 = u64::try_from(dense_values)
        .map_err(|_| InstructionInputSuccessorError::GeometryOverflow)?;
    if dense_values_u64 > u64::from(u32::MAX) + 1 {
        return Err(InstructionInputSuccessorError::ShaderIndexOverflow);
    }
    let dense_table_bytes =
        checked_buffer_bytes(dense_values, FP128_BYTES as usize, max_buffer_length)?;
    Ok(MaterializeShape {
        params: InstructionInputSuccessorMaterializeParams {
            source_elements,
            bound_elements,
            reserved: [0; 2],
        },
        grid_threads: rows / 2,
        resident_row_bytes,
        dense_table_bytes,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseMessageShape {
    params: InstructionInputSuccessorDenseMessageParams,
    grid_threadgroups: usize,
    table_bytes: u64,
    threadgroup_bytes: usize,
}

impl DenseMessageShape {
    pub(crate) const fn params(self) -> InstructionInputSuccessorDenseMessageParams {
        self.params
    }

    pub const fn grid_threadgroups(self) -> usize {
        self.grid_threadgroups
    }

    pub const fn table_bytes(self) -> u64 {
        self.table_bytes
    }

    pub const fn threadgroup_bytes(self) -> usize {
        self.threadgroup_bytes
    }
}

pub fn checked_dense_message_shape(
    table_elements: usize,
    e_in: usize,
    e_out: usize,
    threads_per_threadgroup: usize,
    max_buffer_length: u64,
) -> Result<DenseMessageShape, InstructionInputSuccessorError> {
    let pair_count = e_in
        .checked_mul(e_out)
        .ok_or(InstructionInputSuccessorError::GeometryOverflow)?;
    if table_elements < 2
        || !table_elements.is_power_of_two()
        || e_in == 0
        || e_out == 0
        || pair_count.checked_mul(2) != Some(table_elements)
    {
        return Err(InstructionInputSuccessorError::InvalidEqualitySplit {
            table_elements,
            e_in,
            e_out,
        });
    }
    if threads_per_threadgroup == 0
        || !threads_per_threadgroup.is_multiple_of(INSTRUCTION_INPUT_SUCCESSOR_SIMD_WIDTH)
        || threads_per_threadgroup
            > INSTRUCTION_INPUT_SUCCESSOR_SIMD_WIDTH * INSTRUCTION_INPUT_SUCCESSOR_SIMD_WIDTH
    {
        return Err(InstructionInputSuccessorError::InvalidThreadgroupWidth);
    }
    let table_values = table_elements
        .checked_mul(INSTRUCTION_INPUT_SUCCESSOR_TABLES)
        .ok_or(InstructionInputSuccessorError::GeometryOverflow)?;
    let table_values_u64 = u64::try_from(table_values)
        .map_err(|_| InstructionInputSuccessorError::GeometryOverflow)?;
    if table_values_u64 > u64::from(u32::MAX) + 1 {
        return Err(InstructionInputSuccessorError::ShaderIndexOverflow);
    }
    let table_bytes = checked_buffer_bytes(table_values, FP128_BYTES as usize, max_buffer_length)?;
    let simdgroups = threads_per_threadgroup / INSTRUCTION_INPUT_SUCCESSOR_SIMD_WIDTH;
    let threadgroup_bytes = INSTRUCTION_INPUT_SUCCESSOR_COEFFICIENTS
        .checked_mul(simdgroups)
        .and_then(|values| values.checked_mul(FP128_BYTES as usize))
        .ok_or(InstructionInputSuccessorError::GeometryOverflow)?;
    Ok(DenseMessageShape {
        params: InstructionInputSuccessorDenseMessageParams {
            table_elements: u32::try_from(table_elements)
                .map_err(|_| InstructionInputSuccessorError::ShaderIndexOverflow)?,
            e_in_length: u32::try_from(e_in)
                .map_err(|_| InstructionInputSuccessorError::ShaderIndexOverflow)?,
            e_out_length: u32::try_from(e_out)
                .map_err(|_| InstructionInputSuccessorError::ShaderIndexOverflow)?,
            reserved: 0,
        },
        grid_threadgroups: e_out,
        table_bytes,
        threadgroup_bytes,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhaseWork {
    pub name: &'static str,
    pub core_products: u128,
    pub large_state_bytes: u128,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkPlan {
    pub phases: Vec<PhaseWork>,
}

impl WorkPlan {
    pub fn total_products(&self) -> Result<u128, InstructionInputSuccessorError> {
        self.phases.iter().try_fold(0u128, |total, phase| {
            total
                .checked_add(phase.core_products)
                .ok_or(InstructionInputSuccessorError::GeometryOverflow)
        })
    }

    pub fn total_large_state_bytes(&self) -> Result<u128, InstructionInputSuccessorError> {
        self.phases.iter().try_fold(0u128, |total, phase| {
            total
                .checked_add(phase.large_state_bytes)
                .ok_or(InstructionInputSuccessorError::GeometryOverflow)
        })
    }
}

pub fn current_fused_plan(geometry: Geometry) -> Result<WorkPlan, InstructionInputSuccessorError> {
    let rows = u128::from(geometry.rows);
    let dense_span = geometry.dense_source_span()?;
    Ok(WorkPlan {
        phases: vec![
            PhaseWork {
                name: "native message",
                core_products: checked_mul(rows, 3)?,
                large_state_bytes: checked_mul(rows, 48)?,
            },
            PhaseWork {
                name: "fused native bind and message",
                core_products: checked_mul(rows, 17)? / 2,
                large_state_bytes: checked_mul(rows, 112)?,
            },
            PhaseWork {
                name: "dense ladder",
                core_products: checked_mul(dense_span, 17)? / 2,
                large_state_bytes: checked_mul(dense_span, 192)?,
            },
        ],
    })
}

pub fn split_first_bind_plan(
    geometry: Geometry,
) -> Result<WorkPlan, InstructionInputSuccessorError> {
    let rows = u128::from(geometry.rows);
    let dense_span = geometry.dense_source_span()?;
    Ok(WorkPlan {
        phases: vec![
            PhaseWork {
                name: "native message",
                core_products: checked_mul(rows, 3)?,
                large_state_bytes: checked_mul(rows, 48)?,
            },
            PhaseWork {
                name: "boolean-specialized materialization",
                core_products: checked_mul(rows, 2)?,
                large_state_bytes: checked_mul(rows, 112)?,
            },
            PhaseWork {
                name: "first dense message",
                core_products: checked_mul(rows, 9)? / 2,
                large_state_bytes: checked_mul(rows, 64)?,
            },
            PhaseWork {
                name: "dense ladder",
                core_products: checked_mul(dense_span, 17)? / 2,
                large_state_bytes: checked_mul(dense_span, 192)?,
            },
        ],
    })
}

pub fn split_first_transition_plan(
    geometry: Geometry,
) -> Result<WorkPlan, InstructionInputSuccessorError> {
    let plan = split_first_bind_plan(geometry)?;
    Ok(WorkPlan {
        phases: plan.phases[1..3].to_vec(),
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoofAnchors {
    pub bytes_per_second: u64,
    pub products_per_second: u64,
}

pub fn phase_roof_ns(
    phase: PhaseWork,
    anchors: RoofAnchors,
) -> Result<u128, InstructionInputSuccessorError> {
    if anchors.bytes_per_second == 0 || anchors.products_per_second == 0 {
        return Err(InstructionInputSuccessorError::ZeroRoof);
    }
    let traffic = rate_time_ns(phase.large_state_bytes, anchors.bytes_per_second)?;
    let arithmetic = rate_time_ns(phase.core_products, anchors.products_per_second)?;
    Ok(traffic.max(arithmetic))
}

pub fn sequential_roof_ns(
    plan: &WorkPlan,
    anchors: RoofAnchors,
) -> Result<u128, InstructionInputSuccessorError> {
    plan.phases.iter().try_fold(0u128, |total, phase| {
        total
            .checked_add(phase_roof_ns(*phase, anchors)?)
            .ok_or(InstructionInputSuccessorError::GeometryOverflow)
    })
}

pub fn utilization_cap_ns(
    roof_ns: u128,
    achieved_numerator: u128,
    achieved_denominator: u128,
) -> Result<u128, InstructionInputSuccessorError> {
    if achieved_numerator == 0 || achieved_denominator == 0 {
        return Err(InstructionInputSuccessorError::ZeroRoof);
    }
    ceil_div(
        roof_ns
            .checked_mul(achieved_denominator)
            .ok_or(InstructionInputSuccessorError::GeometryOverflow)?,
        achieved_numerator,
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FactorGate {
    Minimum5x,
    Stretch8x,
}

impl FactorGate {
    pub const fn factor(self) -> u64 {
        match self {
            Self::Minimum5x => 5,
            Self::Stretch8x => 8,
        }
    }

    pub const fn planning_cap_ns(self, cpu_service_ns: u64) -> u64 {
        cpu_service_ns / self.factor()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunOrder {
    CpuFirst,
    MetalFirst,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ServicePair {
    pub cpu_ns: u64,
    pub metal_ns: u64,
    pub order: RunOrder,
}

pub const FROZEN_SERVICE_PAIRS: [ServicePair; 5] = [
    ServicePair {
        cpu_ns: 718_621_795,
        metal_ns: 141_558_454,
        order: RunOrder::CpuFirst,
    },
    ServicePair {
        cpu_ns: 866_175_959,
        metal_ns: 142_462_799,
        order: RunOrder::MetalFirst,
    },
    ServicePair {
        cpu_ns: 731_548_962,
        metal_ns: 154_909_748,
        order: RunOrder::CpuFirst,
    },
    ServicePair {
        cpu_ns: 727_212_419,
        metal_ns: 141_181_207,
        order: RunOrder::MetalFirst,
    },
    ServicePair {
        cpu_ns: 719_473_334,
        metal_ns: 155_366_626,
        order: RunOrder::CpuFirst,
    },
];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PromotionGuards {
    pub exact_round_polynomials: bool,
    pub exact_output_claims: bool,
    pub exact_transcript_and_proof: bool,
    pub source_and_binary_current: bool,
    pub resident_row_identity: bool,
    pub no_round_allocation: bool,
    pub resource_and_spill_capture: bool,
    pub noise_within_limit: bool,
}

impl PromotionGuards {
    pub const fn pass(self) -> bool {
        self.exact_round_polynomials
            && self.exact_output_claims
            && self.exact_transcript_and_proof
            && self.source_and_binary_current
            && self.resident_row_identity
            && self.no_round_allocation
            && self.resource_and_spill_capture
            && self.noise_within_limit
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GateAssessment {
    pub guards_pass: bool,
    pub pooled_median_pass: bool,
    pub cpu_first_median_pass: bool,
    pub metal_first_median_pass: bool,
    pub accepted: bool,
}

pub fn assess_gate(
    gate: FactorGate,
    pairs: &[ServicePair],
    guards: PromotionGuards,
) -> Result<GateAssessment, InstructionInputSuccessorError> {
    let expected_orders = [
        RunOrder::CpuFirst,
        RunOrder::MetalFirst,
        RunOrder::CpuFirst,
        RunOrder::MetalFirst,
        RunOrder::CpuFirst,
    ];
    if pairs.len() != expected_orders.len()
        || !pairs
            .iter()
            .zip(expected_orders)
            .all(|(pair, expected)| pair.order == expected)
    {
        return Err(InstructionInputSuccessorError::InvalidGateTopology);
    }
    if pairs
        .iter()
        .any(|pair| pair.cpu_ns == 0 || pair.metal_ns == 0)
    {
        return Err(InstructionInputSuccessorError::ZeroTimingSample);
    }

    let cpu_first: Vec<_> = pairs
        .iter()
        .copied()
        .filter(|pair| pair.order == RunOrder::CpuFirst)
        .collect();
    let metal_first: Vec<_> = pairs
        .iter()
        .copied()
        .filter(|pair| pair.order == RunOrder::MetalFirst)
        .collect();
    let pooled_median_pass = median_ratio_passes(pairs, gate.factor())?;
    let cpu_first_median_pass = median_ratio_passes(&cpu_first, gate.factor())?;
    let metal_first_median_pass = median_ratio_passes(&metal_first, gate.factor())?;
    let guards_pass = guards.pass();
    Ok(GateAssessment {
        guards_pass,
        pooled_median_pass,
        cpu_first_median_pass,
        metal_first_median_pass,
        accepted: guards_pass
            && pooled_median_pass
            && cpu_first_median_pass
            && metal_first_median_pass,
    })
}

/// Exact crossover test after common later tail work cancels.
pub fn additional_gpu_round_wins(
    gpu_round_ns: u64,
    cpu_round_ns: u64,
    readback_current_ns: u64,
    readback_next_ns: u64,
) -> bool {
    let Some(saved) = cpu_round_ns.checked_add(readback_current_ns) else {
        return false;
    };
    let Some(charged) = gpu_round_ns.checked_add(readback_next_ns) else {
        return false;
    };
    charged < saved
}

fn median_ratio_passes(
    pairs: &[ServicePair],
    factor: u64,
) -> Result<bool, InstructionInputSuccessorError> {
    if pairs.is_empty() || factor == 0 {
        return Err(InstructionInputSuccessorError::InvalidGateTopology);
    }
    let mut sorted = pairs.to_vec();
    sorted.sort_unstable_by(ratio_cmp);
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        return ratio_at_least(sorted[middle], factor);
    }

    let low = sorted[middle - 1];
    let high = sorted[middle];
    let lhs = u128::from(low.cpu_ns)
        .checked_mul(u128::from(high.metal_ns))
        .and_then(|term| {
            u128::from(high.cpu_ns)
                .checked_mul(u128::from(low.metal_ns))
                .and_then(|other| term.checked_add(other))
        })
        .ok_or(InstructionInputSuccessorError::GeometryOverflow)?;
    let rhs = u128::from(low.metal_ns)
        .checked_mul(u128::from(high.metal_ns))
        .and_then(|product| product.checked_mul(u128::from(factor)))
        .and_then(|product| product.checked_mul(2))
        .ok_or(InstructionInputSuccessorError::GeometryOverflow)?;
    Ok(lhs >= rhs)
}

fn ratio_at_least(pair: ServicePair, factor: u64) -> Result<bool, InstructionInputSuccessorError> {
    let charged = u128::from(pair.metal_ns)
        .checked_mul(u128::from(factor))
        .ok_or(InstructionInputSuccessorError::GeometryOverflow)?;
    Ok(u128::from(pair.cpu_ns) >= charged)
}

fn ratio_cmp(lhs: &ServicePair, rhs: &ServicePair) -> Ordering {
    (u128::from(lhs.cpu_ns) * u128::from(rhs.metal_ns))
        .cmp(&(u128::from(rhs.cpu_ns) * u128::from(lhs.metal_ns)))
}

fn checked_buffer_bytes(
    elements: usize,
    element_bytes: usize,
    maximum: u64,
) -> Result<u64, InstructionInputSuccessorError> {
    let requested = elements
        .checked_mul(element_bytes)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(InstructionInputSuccessorError::GeometryOverflow)?;
    if requested > maximum {
        return Err(InstructionInputSuccessorError::BufferTooLong { requested, maximum });
    }
    Ok(requested)
}

fn rate_time_ns(work: u128, rate_per_second: u64) -> Result<u128, InstructionInputSuccessorError> {
    ceil_div(
        work.checked_mul(NANOS_PER_SECOND)
            .ok_or(InstructionInputSuccessorError::GeometryOverflow)?,
        u128::from(rate_per_second),
    )
}

fn ceil_div(numerator: u128, denominator: u128) -> Result<u128, InstructionInputSuccessorError> {
    if denominator == 0 {
        return Err(InstructionInputSuccessorError::ZeroRoof);
    }
    numerator
        .checked_add(denominator - 1)
        .ok_or(InstructionInputSuccessorError::GeometryOverflow)
        .map(|value| value / denominator)
}

fn checked_mul(value: u128, factor: u128) -> Result<u128, InstructionInputSuccessorError> {
    value
        .checked_mul(factor)
        .ok_or(InstructionInputSuccessorError::GeometryOverflow)
}
