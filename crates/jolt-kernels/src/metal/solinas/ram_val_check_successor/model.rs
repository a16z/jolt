//! Exact target-scale work model and fail-closed evidence gates.

use super::abi::{FIELD_BYTES, MESSAGE_COLUMNS, ROW_BYTES, SIMD_WIDTH, STATUS_WORD_BYTES};

pub const TARGET_LOG_T: u32 = 26;
pub const TARGET_LOG_K: u32 = 13;
pub const TARGET_ROWS: u64 = 1 << TARGET_LOG_T;
pub const TARGET_ADDRESSES: u64 = 1 << TARGET_LOG_K;
pub const TARGET_LT_LOW: u64 = 1 << 13;
pub const TARGET_HIGH_BLOCKS: u64 = 1 << 13;
pub const TARGET_GPU_BINDS: u32 = 10;
pub const TARGET_CPU_CUTOFF: u64 = 1 << 16;

pub const SCREENING_EVIDENCE_JSON: &str = include_str!("screening_evidence.json");
pub const FROZEN_CPU_ARTIFACT: &str =
    "benchmark-runs/metal-piop-eval/20260806-133709-697013/result.json";
pub const FROZEN_CPU_ARTIFACT_SHA256: &str =
    "587e00a65bde003a7c3481f58b1ea047ed2c908b0e3d9808bbc7eec6f894b2df";
pub const FROZEN_CPU_REVISION: &str = "5f520c21e338632aa0bf5936ceb02be6c22fa40f";
pub const FROZEN_CPU_SAMPLE_SELECTOR: &str =
    ".attribution_samples[].optimized.kernels[] | select(.kernel == \"RamValCheck\") | .wall_ms";
pub const FROZEN_CPU_SAMPLES_NS: [u64; 5] = [
    240_056_416,
    274_334_163,
    232_004_456,
    234_656_875,
    229_820_624,
];
pub const FROZEN_CPU_MEDIAN_NS: u64 = 234_656_875;
pub const FIVE_X_SCREEN_CAP_NS: u64 = FROZEN_CPU_MEDIAN_NS / 5;
pub const EIGHT_X_SCREEN_CAP_NS: u64 = FROZEN_CPU_MEDIAN_NS / 8;

pub const RETAINED_COPY_BYTES_PER_SECOND: u64 = 451_701_710_520;
pub const RETAINED_FULL_PRODUCTS_PER_SECOND: u64 = 32_330_000_000;
pub const RETAINED_MATCHED_CONTROL_PRODUCTS_PER_SECOND: u64 = 18_100_000_000;
pub const RETAINED_METAL_EVIDENCE: &str =
    "crates/jolt-kernels/autoresearch/evidence/ram_val_check_log26_observed_64c271895.json";
pub const RETAINED_METAL_EVIDENCE_SHA256: &str =
    "2da0a863ab5519c6c9211e7f67bb34bd6f94bf81080203b1ddb016ec3390e9fc";
pub const OBSERVED_DENSE_FIRST_MESSAGE_NS: u64 = 7_918_792;
pub const OBSERVED_HYBRID_NO_FS_MEDIAN_NS: u64 = 31_106_000;
pub const ROOF_EFFICIENCY_PERMILLE: u64 = 800;

/// Activity at or below this value has enough heuristic 8x margin to justify
/// prioritizing a target-scale run after a small parity/proxy run.
pub const SPARSE_SCREEN_TARGET_PRIORITY_PERMILLE: u64 = 600;
/// Above the target-priority region, retain one cheap proxy before spending a
/// target-scale run. Crossing this threshold is not an impossibility result.
pub const SPARSE_SCREEN_PROXY_PRIORITY_PERMILLE: u64 = 680;

const NANOS_PER_SECOND: u128 = 1_000_000_000;
const PERMILLE: u128 = 1_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelError {
    InvalidRows,
    InvalidAddressDomain,
    InvalidFactorization,
    InvalidGpuBindCount,
    InvalidActivity,
    Overflow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub rows: u64,
    pub addresses: u64,
    pub lt_low: u64,
    pub high_blocks: u64,
    pub gpu_binds: u32,
}

impl Geometry {
    pub fn target() -> Self {
        Self {
            rows: TARGET_ROWS,
            addresses: TARGET_ADDRESSES,
            lt_low: TARGET_LT_LOW,
            high_blocks: TARGET_HIGH_BLOCKS,
            gpu_binds: TARGET_GPU_BINDS,
        }
    }

    pub fn validate(self) -> Result<Self, ModelError> {
        if self.rows < 4 || !self.rows.is_power_of_two() || self.rows > u64::from(u32::MAX) {
            return Err(ModelError::InvalidRows);
        }
        if self.addresses == 0
            || !self.addresses.is_power_of_two()
            || self.addresses >= u64::from(u32::MAX)
        {
            return Err(ModelError::InvalidAddressDomain);
        }
        if self.lt_low < 2 * SIMD_WIDTH as u64
            || !self.lt_low.is_power_of_two()
            || !self.high_blocks.is_power_of_two()
            || self.lt_low > u64::from(u32::MAX)
            || self.high_blocks > u64::from(u32::MAX)
            || self.lt_low.checked_mul(self.high_blocks) != Some(self.rows)
        {
            return Err(ModelError::InvalidFactorization);
        }
        let low_rounds = self.lt_low.ilog2();
        if self.gpu_binds == 0 || self.gpu_binds >= low_rounds {
            return Err(ModelError::InvalidGpuBindCount);
        }
        Ok(self)
    }

    pub fn gpu_messages(self) -> u64 {
        u64::from(self.gpu_binds) + 1
    }

    pub fn cpu_cutoff(self) -> u64 {
        self.rows >> self.gpu_binds
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FirstMessageActivity {
    pub active_pairs: u64,
    pub active_simd_iterations: u64,
}

impl FirstMessageActivity {
    pub fn dense(geometry: Geometry) -> Self {
        Self {
            active_pairs: geometry.rows / 2,
            active_simd_iterations: geometry.rows / (2 * SIMD_WIDTH as u64),
        }
    }

    pub fn validate(self, geometry: Geometry) -> Result<Self, ModelError> {
        let geometry = geometry.validate()?;
        let pairs = geometry.rows / 2;
        let iterations = pairs / SIMD_WIDTH as u64;
        let valid_zero = self.active_pairs == 0 && self.active_simd_iterations == 0;
        let valid_nonzero = self.active_pairs > 0
            && self.active_simd_iterations > 0
            && self.active_pairs <= pairs
            && self.active_simd_iterations <= iterations
            && self.active_simd_iterations <= self.active_pairs
            && self.active_pairs
                <= self
                    .active_simd_iterations
                    .checked_mul(SIMD_WIDTH as u64)
                    .ok_or(ModelError::Overflow)?;
        if !valid_zero && !valid_nonzero {
            return Err(ModelError::InvalidActivity);
        }
        Ok(self)
    }

    pub fn active_simd_permille(self, geometry: Geometry) -> Result<u64, ModelError> {
        let geometry = geometry.validate()?;
        let _ = self.validate(geometry)?;
        let total = geometry.rows / (2 * SIMD_WIDTH as u64);
        let scaled = self
            .active_simd_iterations
            .checked_mul(1_000)
            .ok_or(ModelError::Overflow)?;
        Ok(scaled.checked_add(total - 1).ok_or(ModelError::Overflow)? / total)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SuccessorPhase {
    FirstMessage,
    NativeBindAndMessage,
    DenseTransitions,
}

impl SuccessorPhase {
    const ALL: [Self; 3] = [
        Self::FirstMessage,
        Self::NativeBindAndMessage,
        Self::DenseTransitions,
    ];

    const fn index(self) -> usize {
        match self {
            Self::FirstMessage => 0,
            Self::NativeBindAndMessage => 1,
            Self::DenseTransitions => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhaseWork {
    pub name: &'static str,
    /// Source-level field products retained after the exact pair skip.
    pub logical_products: u128,
    /// Products executed only by lane zero after the SIMD reduction.
    pub lane_zero_products: u128,
    /// Full-width-equivalent product slots for the described SIMD32 schedule.
    pub simd_equivalent_product_slots: u128,
    /// Mandatory streaming reads and writes of native or dense state.
    pub large_state_bytes: u128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkPlan {
    pub phases: [PhaseWork; 3],
    pub phase_message_counts: [u64; 3],
    pub phase_bind_write_bytes: [u128; 3],
    pub phase_handoff_bytes: [u128; 3],
    pub partial_per_message_bytes: u128,
    pub partial_global_bytes: u128,
    pub challenge_table_initial_write_bytes: u128,
    pub challenge_table_bind_write_bytes: u128,
    pub cpu_tail_handoff_bytes: u128,
    pub host_message_read_bytes: u128,
    pub status_per_message_bytes: u128,
    pub status_host_io_bytes: u128,
    pub sequence_resident_bytes: u128,
    pub producer_diagnostic_write_bytes: u128,
}

impl WorkPlan {
    pub fn logical_products(&self) -> u128 {
        self.phases.iter().map(|phase| phase.logical_products).sum()
    }

    pub fn lane_zero_products(&self) -> u128 {
        self.phases
            .iter()
            .map(|phase| phase.lane_zero_products)
            .sum()
    }

    pub fn simd_equivalent_product_slots(&self) -> u128 {
        self.phases
            .iter()
            .map(|phase| phase.simd_equivalent_product_slots)
            .sum()
    }

    pub fn large_state_bytes(&self) -> u128 {
        self.phases
            .iter()
            .map(|phase| phase.large_state_bytes)
            .sum()
    }

    pub fn first_message_accounted_bytes(&self) -> Result<u128, ModelError> {
        self.phases[0]
            .large_state_bytes
            .checked_add(self.partial_per_message_bytes)
            .and_then(|value| value.checked_add(self.status_per_message_bytes))
            .ok_or(ModelError::Overflow)
    }

    /// Exact mandatory traffic represented by this schedule. Repeated address
    /// and LT-table cache-line fills are deliberately not invented here.
    pub fn accounted_compulsory_bytes(&self) -> u128 {
        self.large_state_bytes()
            + self.partial_global_bytes
            + self.challenge_table_initial_write_bytes
            + self.challenge_table_bind_write_bytes
            + self.cpu_tail_handoff_bytes
            + self.host_message_read_bytes
            + self.status_host_io_bytes
    }

    pub fn first_message_roof(&self) -> Result<RoofBounds, ModelError> {
        roof_bounds(
            self.phases[0].simd_equivalent_product_slots,
            self.first_message_accounted_bytes()?,
        )
    }

    pub fn phase_accounted_bytes(&self, phase: SuccessorPhase) -> Result<u128, ModelError> {
        let index = phase.index();
        let messages = u128::from(self.phase_message_counts[index]);
        let total_messages = u128::from(self.phase_message_counts.iter().sum::<u64>());
        let host_bytes_per_message = self
            .host_message_read_bytes
            .checked_div(total_messages)
            .ok_or(ModelError::Overflow)?;
        let partial_bytes = self
            .partial_per_message_bytes
            .checked_mul(messages)
            .ok_or(ModelError::Overflow)?;
        let status_bytes = self
            .status_per_message_bytes
            .checked_mul(messages)
            .ok_or(ModelError::Overflow)?;
        let host_bytes = host_bytes_per_message
            .checked_mul(messages)
            .ok_or(ModelError::Overflow)?;
        self.phases[index]
            .large_state_bytes
            .checked_add(partial_bytes)
            .and_then(|value| value.checked_add(status_bytes))
            .and_then(|value| value.checked_add(host_bytes))
            .and_then(|value| value.checked_add(self.phase_bind_write_bytes[index]))
            .and_then(|value| value.checked_add(self.phase_handoff_bytes[index]))
            .ok_or(ModelError::Overflow)
    }

    pub fn phase_roof(&self, phase: SuccessorPhase) -> Result<RoofBounds, ModelError> {
        roof_bounds(
            self.phases[phase.index()].simd_equivalent_product_slots,
            self.phase_accounted_bytes(phase)?,
        )
    }

    pub fn phase_roofs(&self) -> Result<[RoofBounds; 3], ModelError> {
        let [first, native, dense] = SuccessorPhase::ALL;
        Ok([
            self.phase_roof(first)?,
            self.phase_roof(native)?,
            self.phase_roof(dense)?,
        ])
    }

    pub fn prefix_roof(&self) -> Result<RoofBounds, ModelError> {
        roof_bounds(
            self.simd_equivalent_product_slots(),
            self.accounted_compulsory_bytes(),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoofBounds {
    pub compute_floor_ns: u64,
    pub traffic_floor_ns: u64,
    pub optimistic_floor_ns: u64,
    pub eighty_percent_roof_bar_ns: u64,
}

pub fn target_work_plan(activity: FirstMessageActivity) -> Result<WorkPlan, ModelError> {
    work_plan(Geometry::target(), activity)
}

pub fn work_plan(
    geometry: Geometry,
    activity: FirstMessageActivity,
) -> Result<WorkPlan, ModelError> {
    let geometry = geometry.validate()?;
    let activity = activity.validate(geometry)?;
    let rows = u128::from(geometry.rows);
    let high = u128::from(geometry.high_blocks);
    let simd_width = SIMD_WIDTH as u128;

    let first_lane_zero = 6 * high;
    let first_logical = u128::from(activity.active_pairs)
        .checked_mul(6)
        .and_then(|value| value.checked_add(first_lane_zero))
        .ok_or(ModelError::Overflow)?;
    let first_simd_slots = u128::from(activity.active_simd_iterations)
        .checked_mul(6 * simd_width)
        .and_then(|value| value.checked_add(first_lane_zero * simd_width))
        .ok_or(ModelError::Overflow)?;
    let first_bytes = rows.checked_mul(ROW_BYTES).ok_or(ModelError::Overflow)?;

    let native_lane_zero = 6 * high;
    let native_inner = rows
        .checked_mul(5)
        .and_then(|value| value.checked_div(2))
        .ok_or(ModelError::Overflow)?;
    let native_logical = native_inner
        .checked_add(native_lane_zero)
        .ok_or(ModelError::Overflow)?;
    let native_simd_slots = transition_inner_simd_slots(rows, high, simd_width)?
        .checked_add(native_lane_zero * simd_width)
        .ok_or(ModelError::Overflow)?;
    let native_bytes = rows
        .checked_mul(ROW_BYTES)
        .and_then(|read| read.checked_add(rows / 2 * 2 * FIELD_BYTES))
        .ok_or(ModelError::Overflow)?;

    let mut dense_logical = 0_u128;
    let mut dense_lane_zero = 0_u128;
    let mut dense_simd_slots = 0_u128;
    let mut dense_bytes = 0_u128;
    let mut source = rows / 2;
    for _ in 0..geometry.gpu_binds - 1 {
        let inner = 5 * source / 2;
        let lane_zero = 6 * high;
        dense_logical = dense_logical
            .checked_add(inner + lane_zero)
            .ok_or(ModelError::Overflow)?;
        dense_lane_zero = dense_lane_zero
            .checked_add(lane_zero)
            .ok_or(ModelError::Overflow)?;
        let inner_simd_slots = transition_inner_simd_slots(source, high, simd_width)?;
        dense_simd_slots = dense_simd_slots
            .checked_add(inner_simd_slots + lane_zero * simd_width)
            .ok_or(ModelError::Overflow)?;
        dense_bytes = dense_bytes
            .checked_add(source * 2 * FIELD_BYTES + source / 2 * 2 * FIELD_BYTES)
            .ok_or(ModelError::Overflow)?;
        source /= 2;
    }
    if source != u128::from(geometry.cpu_cutoff()) {
        return Err(ModelError::InvalidGpuBindCount);
    }

    let partial_per_message_bytes = reduction_global_bytes(geometry.high_blocks)?;
    let partial_global_bytes = partial_per_message_bytes
        .checked_mul(u128::from(geometry.gpu_messages()))
        .ok_or(ModelError::Overflow)?;
    let challenge_table_initial_write_bytes = u128::from(
        geometry
            .addresses
            .checked_add(geometry.lt_low)
            .and_then(|value| value.checked_add(2 * geometry.high_blocks))
            .ok_or(ModelError::Overflow)?,
    )
    .checked_mul(FIELD_BYTES)
    .ok_or(ModelError::Overflow)?;
    let mut bound_low = geometry.lt_low;
    let mut bound_table_fields = 0_u128;
    let mut first_bind_table_fields = 0_u128;
    for _ in 0..geometry.gpu_binds {
        bound_low /= 2;
        if first_bind_table_fields == 0 {
            first_bind_table_fields = u128::from(bound_low);
        }
        bound_table_fields = bound_table_fields
            .checked_add(u128::from(bound_low))
            .ok_or(ModelError::Overflow)?;
    }
    let challenge_table_bind_write_bytes = bound_table_fields
        .checked_mul(FIELD_BYTES)
        .ok_or(ModelError::Overflow)?;
    let first_bind_write_bytes = first_bind_table_fields
        .checked_mul(FIELD_BYTES)
        .ok_or(ModelError::Overflow)?;
    let dense_bind_write_bytes = challenge_table_bind_write_bytes
        .checked_sub(first_bind_write_bytes)
        .ok_or(ModelError::Overflow)?;
    let cpu_tail_handoff_bytes = u128::from(geometry.cpu_cutoff())
        .checked_mul(2 * FIELD_BYTES)
        .ok_or(ModelError::Overflow)?;
    let host_message_read_bytes = u128::from(geometry.gpu_messages())
        .checked_mul(MESSAGE_COLUMNS as u128 * FIELD_BYTES)
        .ok_or(ModelError::Overflow)?;
    let status_per_message_bytes = 2 * STATUS_WORD_BYTES;
    let status_host_io_bytes = u128::from(geometry.gpu_messages())
        .checked_mul(status_per_message_bytes)
        .ok_or(ModelError::Overflow)?;

    let dense_a = rows / 2 * 2 * FIELD_BYTES;
    let dense_b = rows / 4 * 2 * FIELD_BYTES;
    let table_bytes = challenge_table_initial_write_bytes;
    let partial_buffers = 2 * MESSAGE_COLUMNS as u128 * high * FIELD_BYTES;
    let sequence_resident_bytes = first_bytes
        .checked_add(dense_a)
        .and_then(|value| value.checked_add(dense_b))
        .and_then(|value| value.checked_add(table_bytes))
        .and_then(|value| value.checked_add(partial_buffers))
        .and_then(|value| value.checked_add(STATUS_WORD_BYTES))
        .ok_or(ModelError::Overflow)?;

    Ok(WorkPlan {
        phases: [
            PhaseWork {
                name: "sparse first message",
                logical_products: first_logical,
                lane_zero_products: first_lane_zero,
                simd_equivalent_product_slots: first_simd_slots,
                large_state_bytes: first_bytes,
            },
            PhaseWork {
                name: "native bind and message",
                logical_products: native_logical,
                lane_zero_products: native_lane_zero,
                simd_equivalent_product_slots: native_simd_slots,
                large_state_bytes: native_bytes,
            },
            PhaseWork {
                name: "nine dense transitions",
                logical_products: dense_logical,
                lane_zero_products: dense_lane_zero,
                simd_equivalent_product_slots: dense_simd_slots,
                large_state_bytes: dense_bytes,
            },
        ],
        phase_message_counts: [1, 1, u64::from(geometry.gpu_binds - 1)],
        phase_bind_write_bytes: [0, first_bind_write_bytes, dense_bind_write_bytes],
        phase_handoff_bytes: [0, 0, cpu_tail_handoff_bytes],
        partial_per_message_bytes,
        partial_global_bytes,
        challenge_table_initial_write_bytes,
        challenge_table_bind_write_bytes,
        cpu_tail_handoff_bytes,
        host_message_read_bytes,
        status_per_message_bytes,
        status_host_io_bytes,
        sequence_resident_bytes,
        producer_diagnostic_write_bytes: first_bytes,
    })
}

/// A transition performs ten products per low pair. Once a high block has
/// fewer than 32 pairs, masked lanes still occupy SIMD32 product slots.
fn transition_inner_simd_slots(
    source_rows: u128,
    high_blocks: u128,
    simd_width: u128,
) -> Result<u128, ModelError> {
    let rows_per_iteration = high_blocks.checked_mul(4).ok_or(ModelError::Overflow)?;
    if source_rows % rows_per_iteration != 0 {
        return Err(ModelError::InvalidFactorization);
    }
    let low_pairs_per_high = source_rows / rows_per_iteration;
    let iterations = low_pairs_per_high.div_ceil(simd_width);
    iterations
        .checked_mul(10 * simd_width)
        .and_then(|value| value.checked_mul(high_blocks))
        .ok_or(ModelError::Overflow)
}

fn reduction_global_bytes(input_count: u64) -> Result<u128, ModelError> {
    let columns = MESSAGE_COLUMNS as u128;
    let mut count = u128::from(input_count);
    let mut bytes = columns
        .checked_mul(count)
        .and_then(|fields| fields.checked_mul(FIELD_BYTES))
        .ok_or(ModelError::Overflow)?;
    while count > 1 {
        let output = count.div_ceil(SIMD_WIDTH as u128);
        bytes = bytes
            .checked_add(columns * count * FIELD_BYTES)
            .and_then(|value| value.checked_add(columns * output * FIELD_BYTES))
            .ok_or(ModelError::Overflow)?;
        count = output;
    }
    Ok(bytes)
}

pub fn product_floor_ns(product_slots: u128) -> Result<u64, ModelError> {
    rate_floor_ns(product_slots, RETAINED_FULL_PRODUCTS_PER_SECOND)
}

pub fn transfer_floor_ns(bytes: u128) -> Result<u64, ModelError> {
    rate_floor_ns(bytes, RETAINED_COPY_BYTES_PER_SECOND)
}

fn rate_floor_ns(units: u128, units_per_second: u64) -> Result<u64, ModelError> {
    let numerator = units
        .checked_mul(NANOS_PER_SECOND)
        .and_then(|value| value.checked_add(u128::from(units_per_second) - 1))
        .ok_or(ModelError::Overflow)?;
    u64::try_from(numerator / u128::from(units_per_second)).map_err(|_| ModelError::Overflow)
}

fn roof_bounds(product_slots: u128, bytes: u128) -> Result<RoofBounds, ModelError> {
    let compute_floor_ns = product_floor_ns(product_slots)?;
    let traffic_floor_ns = transfer_floor_ns(bytes)?;
    let optimistic_floor_ns = compute_floor_ns.max(traffic_floor_ns);
    let bar_numerator = u128::from(optimistic_floor_ns)
        .checked_mul(PERMILLE)
        .and_then(|value| value.checked_add(u128::from(ROOF_EFFICIENCY_PERMILLE) - 1))
        .ok_or(ModelError::Overflow)?;
    let eighty_percent_roof_bar_ns =
        u64::try_from(bar_numerator / u128::from(ROOF_EFFICIENCY_PERMILLE))
            .map_err(|_| ModelError::Overflow)?;
    Ok(RoofBounds {
        compute_floor_ns,
        traffic_floor_ns,
        optimistic_floor_ns,
        eighty_percent_roof_bar_ns,
    })
}

/// Historical interpolation used only to rank experiments. It is neither a
/// lower nor an upper bound and therefore cannot reject a mechanism.
pub fn heuristic_first_message_ns(active_simd_permille: u64) -> Result<u64, ModelError> {
    if active_simd_permille > 1_000 {
        return Err(ModelError::InvalidActivity);
    }
    let row_floor = transfer_floor_ns(u128::from(TARGET_ROWS) * ROW_BYTES)?;
    let scalable = OBSERVED_DENSE_FIRST_MESSAGE_NS
        .checked_sub(row_floor)
        .ok_or(ModelError::Overflow)?;
    let scaled = u128::from(scalable)
        .checked_mul(u128::from(active_simd_permille))
        .and_then(|value| value.checked_add(PERMILLE - 1))
        .ok_or(ModelError::Overflow)?
        / PERMILLE;
    u64::try_from(u128::from(row_floor) + scaled).map_err(|_| ModelError::Overflow)
}

pub fn heuristic_hybrid_ns(active_simd_permille: u64) -> Result<u64, ModelError> {
    let fixed_observed = OBSERVED_HYBRID_NO_FS_MEDIAN_NS
        .checked_sub(OBSERVED_DENSE_FIRST_MESSAGE_NS)
        .ok_or(ModelError::Overflow)?;
    fixed_observed
        .checked_add(heuristic_first_message_ns(active_simd_permille)?)
        .ok_or(ModelError::Overflow)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseScreenClass {
    TargetScalePriority,
    ProxyFirst,
    LowPriority,
}

pub fn sparse_screen_class(active_simd_permille: u64) -> Result<SparseScreenClass, ModelError> {
    if active_simd_permille > 1_000 {
        return Err(ModelError::InvalidActivity);
    }
    if active_simd_permille <= SPARSE_SCREEN_TARGET_PRIORITY_PERMILLE {
        Ok(SparseScreenClass::TargetScalePriority)
    } else if active_simd_permille < SPARSE_SCREEN_PROXY_PRIORITY_PERMILLE {
        Ok(SparseScreenClass::ProxyFirst)
    } else {
        Ok(SparseScreenClass::LowPriority)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProducerKind {
    SharedIncrementAccess,
    DedicatedRamValPack,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProducerEvidence {
    pub kind: ProducerKind,
    pub rows: u64,
    pub row_bytes: u64,
    pub allocations: u64,
    pub rows_written: u64,
    pub row_upload_bytes: u64,
    pub full_domain_copy_bytes: u64,
    pub full_domain_temporary_row_bytes: u64,
    pub streaming_scratch_peak_bytes: u64,
    pub prepare_storage_id: u64,
    pub ram_val_storage_id: u64,
    pub terminal_storage_id: u64,
    pub produced_before_piop: bool,
    pub retained_through_stage7: bool,
    pub semantics_checked: bool,
    pub active_pairs: u64,
    pub active_simd_iterations: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProducerRejection {
    WrongKind,
    WrongShape,
    RepackedOrCopied,
    MissingIdentity,
    IdentityChanged,
    WrongLifetime,
    SemanticsUnchecked,
    InvalidActivity,
}

impl ProducerEvidence {
    pub fn validate(self) -> Result<FirstMessageActivity, ProducerRejection> {
        if self.kind != ProducerKind::SharedIncrementAccess {
            return Err(ProducerRejection::WrongKind);
        }
        if self.rows != TARGET_ROWS
            || self.row_bytes != ROW_BYTES as u64
            || self.allocations != 1
            || self.rows_written != TARGET_ROWS
        {
            return Err(ProducerRejection::WrongShape);
        }
        if self.row_upload_bytes != 0
            || self.full_domain_copy_bytes != 0
            || self.full_domain_temporary_row_bytes != 0
            || self.streaming_scratch_peak_bytes >= TARGET_ROWS * ROW_BYTES as u64
        {
            return Err(ProducerRejection::RepackedOrCopied);
        }
        if self.prepare_storage_id == 0
            || self.ram_val_storage_id == 0
            || self.terminal_storage_id == 0
        {
            return Err(ProducerRejection::MissingIdentity);
        }
        if self.prepare_storage_id != self.ram_val_storage_id
            || self.prepare_storage_id != self.terminal_storage_id
        {
            return Err(ProducerRejection::IdentityChanged);
        }
        if !self.produced_before_piop || !self.retained_through_stage7 {
            return Err(ProducerRejection::WrongLifetime);
        }
        if !self.semantics_checked {
            return Err(ProducerRejection::SemanticsUnchecked);
        }
        FirstMessageActivity {
            active_pairs: self.active_pairs,
            active_simd_iterations: self.active_simd_iterations,
        }
        .validate(Geometry::target())
        .map_err(|_| ProducerRejection::InvalidActivity)
    }
}

/// Counters and identities that bind the sparse work plan to one producer run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActivityProvenance {
    pub source_revision: [u8; 20],
    pub artifact_sha256: [u8; 32],
    pub trace_sha256: [u8; 32],
    pub storage_id: u64,
    pub rows: u64,
    pub active_pairs: u64,
    pub active_simd_iterations: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivityProvenanceRejection {
    MissingSourceRevision,
    WrongSourceRevision,
    MissingArtifactHash,
    MissingTraceHash,
    WrongStorageIdentity,
    WrongRows,
    WrongActivity,
}

impl ActivityProvenance {
    pub fn validate(
        self,
        producer: ProducerEvidence,
        expected_revision: [u8; 20],
    ) -> Result<(), ActivityProvenanceRejection> {
        if is_zero_digest(&self.source_revision) {
            return Err(ActivityProvenanceRejection::MissingSourceRevision);
        }
        if self.source_revision != expected_revision {
            return Err(ActivityProvenanceRejection::WrongSourceRevision);
        }
        if is_zero_digest(&self.artifact_sha256) {
            return Err(ActivityProvenanceRejection::MissingArtifactHash);
        }
        if is_zero_digest(&self.trace_sha256) {
            return Err(ActivityProvenanceRejection::MissingTraceHash);
        }
        if self.storage_id == 0 || self.storage_id != producer.prepare_storage_id {
            return Err(ActivityProvenanceRejection::WrongStorageIdentity);
        }
        if self.rows != producer.rows {
            return Err(ActivityProvenanceRejection::WrongRows);
        }
        if self.active_pairs != producer.active_pairs
            || self.active_simd_iterations != producer.active_simd_iterations
        {
            return Err(ActivityProvenanceRejection::WrongActivity);
        }
        Ok(())
    }
}

/// Five exact-boundary latency measurements for each admission phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhaseLatencySamples {
    pub first_message_ns: [u64; 5],
    pub native_bind_and_message_ns: [u64; 5],
    pub dense_transitions_ns: [u64; 5],
}

impl PhaseLatencySamples {
    const fn for_phase(self, phase: SuccessorPhase) -> [u64; 5] {
        match phase {
            SuccessorPhase::FirstMessage => self.first_message_ns,
            SuccessorPhase::NativeBindAndMessage => self.native_bind_and_message_ns,
            SuccessorPhase::DenseTransitions => self.dense_transitions_ns,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhaseRoofRejection {
    MissingSamples {
        phase: SuccessorPhase,
    },
    ExceedsBar {
        phase: SuccessorPhase,
        median_ns: u64,
        bar_ns: u64,
    },
}

/// Compiled resource capture for one complete phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompiledPhaseResources {
    pub allocated_registers_per_thread: u32,
    pub resident_simdgroups_per_core: u32,
    pub required_resident_simdgroups_per_core: u32,
    pub spill_bytes: u64,
}

/// Same-revision hashes and resource captures for all admission phases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompiledCaptureEvidence {
    pub source_revision: [u8; 20],
    pub binary_sha256: [u8; 32],
    pub capture_sha256: [u8; 32],
    pub phases: [CompiledPhaseResources; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompiledCaptureRejection {
    MissingSourceRevision,
    WrongSourceRevision,
    MissingBinaryHash,
    MissingCaptureHash,
    MissingRegisterCount {
        phase: SuccessorPhase,
    },
    MissingResidencyRequirement {
        phase: SuccessorPhase,
    },
    MissingResidencyCapture {
        phase: SuccessorPhase,
    },
    ResidencyBelowRequirement {
        phase: SuccessorPhase,
        required: u32,
        got: u32,
    },
    DeviceMemorySpill {
        phase: SuccessorPhase,
        bytes: u64,
    },
}

impl CompiledCaptureEvidence {
    pub fn validate(self, expected_revision: [u8; 20]) -> Result<(), CompiledCaptureRejection> {
        if is_zero_digest(&self.source_revision) {
            return Err(CompiledCaptureRejection::MissingSourceRevision);
        }
        if self.source_revision != expected_revision {
            return Err(CompiledCaptureRejection::WrongSourceRevision);
        }
        if is_zero_digest(&self.binary_sha256) {
            return Err(CompiledCaptureRejection::MissingBinaryHash);
        }
        if is_zero_digest(&self.capture_sha256) {
            return Err(CompiledCaptureRejection::MissingCaptureHash);
        }
        for phase in SuccessorPhase::ALL {
            let resources = self.phases[phase.index()];
            if resources.allocated_registers_per_thread == 0 {
                return Err(CompiledCaptureRejection::MissingRegisterCount { phase });
            }
            if resources.required_resident_simdgroups_per_core == 0 {
                return Err(CompiledCaptureRejection::MissingResidencyRequirement { phase });
            }
            if resources.resident_simdgroups_per_core == 0 {
                return Err(CompiledCaptureRejection::MissingResidencyCapture { phase });
            }
            if resources.resident_simdgroups_per_core
                < resources.required_resident_simdgroups_per_core
            {
                return Err(CompiledCaptureRejection::ResidencyBelowRequirement {
                    phase,
                    required: resources.required_resident_simdgroups_per_core,
                    got: resources.resident_simdgroups_per_core,
                });
            }
            if resources.spill_bytes != 0 {
                return Err(CompiledCaptureRejection::DeviceMemorySpill {
                    phase,
                    bytes: resources.spill_bytes,
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CandidateEvidence {
    pub producer: Option<ProducerEvidence>,
    pub independent_oracle_parity: bool,
    pub output_claim_parity: bool,
    pub clear_and_zk_proofs_verified: bool,
    pub host_fiat_shamir_preserved: bool,
    pub same_artifact_boundary: bool,
    pub alternating_pair_order: bool,
    pub paired_artifact_provenance_recorded: bool,
    pub paired_piop_validation: bool,
    pub paired_cpu_samples_ns: Option<[u64; 5]>,
    pub paired_candidate_samples_ns: Option<[u64; 5]>,
    pub paired_source_revision: [u8; 20],
    pub activity_provenance: Option<ActivityProvenance>,
    pub phase_latency_samples: Option<PhaseLatencySamples>,
    pub compiled_capture: Option<CompiledCaptureEvidence>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionDecision {
    RejectMissingProducer,
    RejectProducer(ProducerRejection),
    RejectCorrectness,
    RejectEvidenceContract,
    NeedsIntegratedMeasurement,
    RejectBelowFiveX,
    RejectMissingActivityProvenance,
    RejectActivityProvenance(ActivityProvenanceRejection),
    RejectMissingPhaseLatencyEvidence,
    RejectPhaseRoof(PhaseRoofRejection),
    RejectMissingCompiledCapture,
    RejectCompiledCapture(CompiledCaptureRejection),
    RejectModel(ModelError),
    PassFiveXPursueEightX,
    PassEightX,
}

/// Ranks a same-revision paired result. A passing screen is not promotion.
pub fn speed_screen_decision(evidence: CandidateEvidence) -> AdmissionDecision {
    let Some(producer) = evidence.producer else {
        return AdmissionDecision::RejectMissingProducer;
    };
    if let Err(rejection) = producer.validate() {
        return AdmissionDecision::RejectProducer(rejection);
    }
    if !evidence.independent_oracle_parity
        || !evidence.output_claim_parity
        || !evidence.clear_and_zk_proofs_verified
        || !evidence.host_fiat_shamir_preserved
    {
        return AdmissionDecision::RejectCorrectness;
    }
    if !evidence.same_artifact_boundary
        || !evidence.alternating_pair_order
        || !evidence.paired_artifact_provenance_recorded
    {
        return AdmissionDecision::RejectEvidenceContract;
    }
    if !evidence.paired_piop_validation {
        return AdmissionDecision::NeedsIntegratedMeasurement;
    }
    let (Some(cpu_samples), Some(candidate_samples)) = (
        evidence.paired_cpu_samples_ns,
        evidence.paired_candidate_samples_ns,
    ) else {
        return AdmissionDecision::NeedsIntegratedMeasurement;
    };
    let (Some(cpu_median), Some(candidate_median)) = (
        median_of_five(cpu_samples),
        median_of_five(candidate_samples),
    ) else {
        return AdmissionDecision::NeedsIntegratedMeasurement;
    };
    if !speedup_at_least(cpu_median, candidate_median, 5) {
        AdmissionDecision::RejectBelowFiveX
    } else if !speedup_at_least(cpu_median, candidate_median, 8) {
        AdmissionDecision::PassFiveXPursueEightX
    } else {
        AdmissionDecision::PassEightX
    }
}

/// Applies the complete promotion contract after the paired speed screen.
pub fn admission_decision(evidence: CandidateEvidence) -> AdmissionDecision {
    let screen = speed_screen_decision(evidence);
    if !matches!(
        screen,
        AdmissionDecision::PassFiveXPursueEightX | AdmissionDecision::PassEightX
    ) {
        return screen;
    }
    if is_zero_digest(&evidence.paired_source_revision) {
        return AdmissionDecision::RejectEvidenceContract;
    }
    let Some(producer) = evidence.producer else {
        return AdmissionDecision::RejectMissingProducer;
    };
    let Some(activity_provenance) = evidence.activity_provenance else {
        return AdmissionDecision::RejectMissingActivityProvenance;
    };
    if let Err(rejection) = activity_provenance.validate(producer, evidence.paired_source_revision)
    {
        return AdmissionDecision::RejectActivityProvenance(rejection);
    }
    let activity = match producer.validate() {
        Ok(activity) => activity,
        Err(rejection) => return AdmissionDecision::RejectProducer(rejection),
    };
    let plan = match target_work_plan(activity) {
        Ok(plan) => plan,
        Err(error) => return AdmissionDecision::RejectModel(error),
    };
    let Some(phase_latency_samples) = evidence.phase_latency_samples else {
        return AdmissionDecision::RejectMissingPhaseLatencyEvidence;
    };
    let roofs = match plan.phase_roofs() {
        Ok(roofs) => roofs,
        Err(error) => return AdmissionDecision::RejectModel(error),
    };
    for phase in SuccessorPhase::ALL {
        let Some(median_ns) = median_of_five(phase_latency_samples.for_phase(phase)) else {
            return AdmissionDecision::RejectPhaseRoof(PhaseRoofRejection::MissingSamples {
                phase,
            });
        };
        let bar_ns = roofs[phase.index()].eighty_percent_roof_bar_ns;
        if median_ns > bar_ns {
            return AdmissionDecision::RejectPhaseRoof(PhaseRoofRejection::ExceedsBar {
                phase,
                median_ns,
                bar_ns,
            });
        }
    }
    let Some(compiled_capture) = evidence.compiled_capture else {
        return AdmissionDecision::RejectMissingCompiledCapture;
    };
    if let Err(rejection) = compiled_capture.validate(evidence.paired_source_revision) {
        return AdmissionDecision::RejectCompiledCapture(rejection);
    }
    screen
}

fn is_zero_digest<const N: usize>(digest: &[u8; N]) -> bool {
    digest.iter().all(|byte| *byte == 0)
}

pub fn median_of_five(mut samples: [u64; 5]) -> Option<u64> {
    if samples.contains(&0) {
        return None;
    }
    samples.sort_unstable();
    Some(samples[2])
}

fn speedup_at_least(cpu_ns: u64, candidate_ns: u64, multiple: u64) -> bool {
    candidate_ns
        .checked_mul(multiple)
        .is_some_and(|threshold| cpu_ns >= threshold)
}
