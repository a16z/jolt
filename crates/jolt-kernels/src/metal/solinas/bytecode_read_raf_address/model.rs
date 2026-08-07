//! Exact work, traffic, and receipt-bound roof model for the tiled worker.

use core::fmt;

use super::carrier::{
    AddressMajorShape, CarrierError, TopologyScheduleReceipt, ValidatedAddressMajorCarrier,
    CELL_BYTES, INNER_SIGN_BYTES, MAGNITUDE_BYTES, RESIDENT_ROW_BYTES, SIMD_WIDTH,
};

pub const STAGES: u64 = 9;
pub const BASE_STAGES: u64 = 5;
pub const FUSED_STAGES: u64 = 4;
pub const FIELD_BYTES: u64 = 16;

pub const CPU_LOG26_MEDIAN_NS: u64 = 204_085_127;
pub const STANDALONE_FIVE_X_CAP_NS: u64 = CPU_LOG26_MEDIAN_NS / 5;
pub const ADDRESS_CYCLE_CPU_LOG26_MEDIAN_NS: u64 = 1_227_486_958;
pub const CYCLE_METAL_LOG26_MEDIAN_NS: u64 = 147_619_957;
pub const DERIVED_FAMILY_SEVEN_X_ADDRESS_NS: u64 =
    ADDRESS_CYCLE_CPU_LOG26_MEDIAN_NS / 7 - CYCLE_METAL_LOG26_MEDIAN_NS;
pub const HARD_LOG26_TARGET_NS: u64 = 27_700_000;

pub const HOST_ROUND_PROXY_NS: u64 = 7_918_251;
pub const COMMAND_BOUNDARY_NS: u64 = 141_000;
pub const COPY_BYTES_PER_SECOND: u64 = 451_701_710_520;
pub const FULL_FIELD_CONSERVATIVE_PRODUCTS_PER_SECOND: u64 = 18_100_000_000;
pub const SIGNED_U64_ADMISSION_TERMS_PER_SECOND: u64 = 26_272_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeasuredTopology {
    pub short_occurrences: u64,
    pub long_occurrences: u64,
    pub short_runs: u64,
    pub long_runs: u64,
    pub maximum_run: u64,
}

impl MeasuredTopology {
    pub const FIBONACCI_LOG26: Self = Self {
        short_occurrences: 1_239,
        long_occurrences: 67_107_625,
        short_runs: 1_059,
        long_runs: 18_949,
        maximum_run: 32_768,
    };

    pub fn validate(self, shape: AddressMajorShape) -> Result<(), ModelError> {
        let rows = as_u64("rows", shape.rows()?)?;
        let outer = as_u64("outer blocks", shape.outer_length()?)?;
        let cells = as_u64("cells", shape.cells()?)?;
        let inner = as_u64("inner length", shape.inner_length()?)?;
        let runs = add("runs", self.short_runs, self.long_runs)?;
        if add(
            "measured occurrences",
            self.short_occurrences,
            self.long_occurrences,
        )? != rows
            || runs < outer
            || runs > cells
            || self.short_occurrences < self.short_runs
            || self.short_occurrences > mul("short occurrences", 32, self.short_runs)?
            || self.long_occurrences < mul("long occurrences", 33, self.long_runs)?
            || self.long_occurrences > mul("long occurrences", inner, self.long_runs)?
            || self.maximum_run <= 32
            || self.maximum_run > inner
            || self.long_occurrences
                > mul("maximum-run coverage", self.maximum_run, self.long_runs)?
        {
            return Err(ModelError::InvalidTopology);
        }
        Ok(())
    }

    pub fn runs(self) -> Result<u64, ModelError> {
        add("runs", self.short_runs, self.long_runs)
    }

    /// Honest schedule bounds when the aggregate CSR census lacks padding counters.
    pub fn schedule_bounds(self, shape: AddressMajorShape) -> Result<ScheduleBounds, ModelError> {
        self.validate(shape)?;
        let simd = SIMD_WIDTH as u64;
        let short_padding = mul("short run padding", simd, self.short_runs)?;
        let minimum_long_padding = round_up("minimum long padding", self.long_occurrences, simd)?;
        let maximum_long_padding = add(
            "maximum long padding",
            self.long_occurrences,
            mul("long tail lanes", simd - 1, self.long_runs)?,
        )? / simd
            * simd;

        let optimistic = self.with_schedule(short_padding, minimum_long_padding);
        let pessimistic = self.with_schedule(short_padding, maximum_long_padding);
        optimistic.validate(shape)?;
        pessimistic.validate(shape)?;
        Ok(ScheduleBounds {
            optimistic,
            pessimistic,
        })
    }

    const fn with_schedule(
        self,
        padded_short_lanes: u64,
        padded_long_lanes: u64,
    ) -> TopologyScheduleReceipt {
        TopologyScheduleReceipt {
            short_occurrences: self.short_occurrences,
            long_occurrences: self.long_occurrences,
            short_runs: self.short_runs,
            long_runs: self.long_runs,
            padded_short_lanes,
            padded_long_lanes,
            maximum_run: self.maximum_run,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScheduleBounds {
    pub optimistic: TopologyScheduleReceipt,
    pub pessimistic: TopologyScheduleReceipt,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StageTiling {
    FiveThenFour,
    NineSinglePass,
}

impl StageTiling {
    pub const fn occurrence_passes(self) -> u64 {
        match self {
            Self::FiveThenFour => 2,
            Self::NineSinglePass => 1,
        }
    }

    pub const fn compact_bytes_per_row(self) -> u64 {
        match self {
            Self::FiveThenFour => 16,
            Self::NineSinglePass => 12,
        }
    }

    pub const fn structural_accumulators(self) -> u64 {
        match self {
            Self::FiveThenFour => 5,
            Self::NineSinglePass => 9,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryFootprint {
    pub shared_rows: u64,
    pub packed_cells: u64,
    pub inner_sign: u64,
    pub magnitude: u64,
    pub equality_lo: u64,
    pub equality_hi: u64,
    pub partials: u64,
    pub pushforwards: u64,
    pub carrier_and_worker_owned: u64,
    pub aggregate_with_shared_rows: u64,
}

pub fn memory_footprint(
    shape: AddressMajorShape,
    outer_tiles: u64,
) -> Result<MemoryFootprint, ModelError> {
    let rows = as_u64("rows", shape.rows()?)?;
    let cells = as_u64("cells", shape.cells()?)?;
    let inner = as_u64("inner length", shape.inner_length()?)?;
    let outer = as_u64("outer length", shape.outer_length()?)?;
    let addresses = as_u64("addresses", shape.addresses()?)?;
    validate_outer_tiles(outer, outer_tiles)?;
    let shared_rows = mul("shared rows", RESIDENT_ROW_BYTES as u64, rows)?;
    let packed_cells = mul("packed cells", CELL_BYTES as u64, cells)?;
    let inner_sign = mul("inner/sign stream", INNER_SIGN_BYTES as u64, rows)?;
    let magnitude = mul("magnitude stream", MAGNITUDE_BYTES as u64, rows)?;
    let equality_lo = mul(
        "low equality tables",
        mul("stage fields", FIELD_BYTES, STAGES)?,
        inner,
    )?;
    let equality_hi = mul(
        "high equality tables",
        mul("stage fields", FIELD_BYTES, STAGES)?,
        outer,
    )?;
    let partials = mul(
        "tile partials",
        mul(
            "tile partial fields",
            mul("stage addresses", STAGES, addresses)?,
            outer_tiles,
        )?,
        FIELD_BYTES,
    )?;
    let pushforwards = mul(
        "pushforwards",
        mul("stage fields", FIELD_BYTES, STAGES)?,
        addresses,
    )?;
    let carrier_and_worker_owned = sum(&[
        packed_cells,
        inner_sign,
        magnitude,
        equality_lo,
        equality_hi,
        partials,
        pushforwards,
    ])?;
    Ok(MemoryFootprint {
        shared_rows,
        packed_cells,
        inner_sign,
        magnitude,
        equality_lo,
        equality_hi,
        partials,
        pushforwards,
        carrier_and_worker_owned,
        aggregate_with_shared_rows: add(
            "aggregate storage",
            shared_rows,
            carrier_and_worker_owned,
        )?,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Work {
    pub useful_base_updates: u64,
    pub useful_fused_updates: u64,
    pub useful_outer_products: u64,
    pub issued_base_update_lanes: u64,
    pub issued_fused_update_lanes: u64,
    pub issued_outer_product_lanes: u64,
    pub host_equality_generation_products: u64,
    pub simd_run_reduction_lanes: u64,
    pub threadgroup_reduction_lanes: u64,
    pub tile_reduction_lanes: u64,
    pub issued_reduction_addition_lanes: u64,
    pub producer_count_cursor_atomics: u64,
    pub member_output_atomics: u64,
}

pub fn work(
    shape: AddressMajorShape,
    topology: TopologyScheduleReceipt,
    outer_tiles: u64,
) -> Result<Work, ModelError> {
    topology.validate(shape)?;
    let rows = as_u64("rows", shape.rows()?)?;
    let addresses = as_u64("addresses", shape.addresses()?)?;
    let inner = as_u64("inner length", shape.inner_length()?)?;
    let outer = as_u64("outer length", shape.outer_length()?)?;
    validate_outer_tiles(outer, outer_tiles)?;
    let padded = topology.padded_lanes()?;
    let runs = topology.runs()?;
    let issued_outer_product_lanes =
        mul("issued outer products", STAGES * SIMD_WIDTH as u64, runs)?;
    let equality_nodes = add(
        "equality nodes",
        inner
            .checked_sub(1)
            .ok_or(ModelError::Overflow("low equality nodes"))?,
        outer
            .checked_sub(1)
            .ok_or(ModelError::Overflow("high equality nodes"))?,
    )?;
    let simd_run_reduction_lanes = mul("run reductions", 1_440, runs)?;
    let threadgroup_reduction_lanes = mul(
        "threadgroup reductions",
        512,
        mul("address tile groups", addresses, outer_tiles)?,
    )?;
    let tile_reduction_lanes = mul(
        "tile reductions",
        STAGES,
        mul("address tile fields", addresses, outer_tiles)?,
    )?;
    Ok(Work {
        useful_base_updates: mul("useful base updates", BASE_STAGES, rows)?,
        useful_fused_updates: mul("useful fused updates", FUSED_STAGES, rows)?,
        useful_outer_products: mul("useful outer products", STAGES, runs)?,
        issued_base_update_lanes: mul("issued base updates", BASE_STAGES, padded)?,
        issued_fused_update_lanes: mul("issued fused updates", FUSED_STAGES, padded)?,
        issued_outer_product_lanes,
        host_equality_generation_products: mul(
            "host equality products",
            2 * STAGES,
            equality_nodes,
        )?,
        simd_run_reduction_lanes,
        threadgroup_reduction_lanes,
        tile_reduction_lanes,
        issued_reduction_addition_lanes: sum(&[
            simd_run_reduction_lanes,
            threadgroup_reduction_lanes,
            tile_reduction_lanes,
        ])?,
        producer_count_cursor_atomics: mul("producer atomics", 2, rows)?,
        member_output_atomics: 0,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Traffic {
    pub producer_cursor_requested: u64,
    pub producer_rank_scatter_requested: u64,
    pub producer_rank_whole_piop_requested: u64,
    pub member_source_row_bytes: u64,
    pub packed_cell_bytes: u64,
    pub packed_cell_read_bytes: u64,
    pub compact_read_bytes: u64,
    pub equality_lo_requested: u64,
    pub equality_hi_requested: u64,
    pub partial_write_read_bytes: u64,
    pub output_write_bytes: u64,
    pub worker_forced_streaming_floor: u64,
    pub worker_physical_unique_minimum: u64,
    pub worker_shader_requested: u64,
}

pub fn traffic(
    shape: AddressMajorShape,
    topology: TopologyScheduleReceipt,
    tiling: StageTiling,
    outer_tiles: u64,
) -> Result<Traffic, ModelError> {
    topology.validate(shape)?;
    let rows = as_u64("rows", shape.rows()?)?;
    let cells = as_u64("cells", shape.cells()?)?;
    let addresses = as_u64("addresses", shape.addresses()?)?;
    let outer = as_u64("outer length", shape.outer_length()?)?;
    validate_outer_tiles(outer, outer_tiles)?;
    let memory = memory_footprint(shape, outer_tiles)?;
    let packed_cell_bytes = mul("packed-cell bytes", CELL_BYTES as u64, cells)?;
    let packed_cell_read_bytes = mul(
        "packed-cell reads",
        tiling.occurrence_passes(),
        packed_cell_bytes,
    )?;
    let compact_read_bytes = mul("compact reads", tiling.compact_bytes_per_row(), rows)?;
    let equality_lo_requested = mul("E_lo requests", FIELD_BYTES * STAGES, rows)?;
    let equality_hi_requested = mul("E_hi requests", FIELD_BYTES * STAGES, topology.runs()?)?;
    let partial_write_read_bytes = mul("partial write/read", 2, memory.partials)?;
    let output_write_bytes = mul("output writes", FIELD_BYTES * STAGES, addresses)?;
    let worker_forced_streaming_floor = sum(&[
        packed_cell_read_bytes,
        compact_read_bytes,
        memory.equality_lo,
        memory.equality_hi,
        partial_write_read_bytes,
        output_write_bytes,
    ])?;
    Ok(Traffic {
        producer_cursor_requested: add(
            "cursor producer bytes",
            mul("producer row bytes", 28, rows)?,
            packed_cell_bytes,
        )?,
        producer_rank_scatter_requested: add(
            "rank scatter bytes",
            mul("rank scatter row bytes", 30, rows)?,
            packed_cell_bytes,
        )?,
        producer_rank_whole_piop_requested: add(
            "rank whole-PIOP bytes",
            mul("rank whole-PIOP row bytes", 32, rows)?,
            packed_cell_bytes,
        )?,
        member_source_row_bytes: 0,
        packed_cell_bytes,
        packed_cell_read_bytes,
        compact_read_bytes,
        equality_lo_requested,
        equality_hi_requested,
        partial_write_read_bytes,
        output_write_bytes,
        worker_forced_streaming_floor,
        worker_physical_unique_minimum: memory.carrier_and_worker_owned,
        worker_shader_requested: sum(&[
            packed_cell_read_bytes,
            compact_read_bytes,
            equality_lo_requested,
            equality_hi_requested,
            partial_write_read_bytes,
            output_write_bytes,
        ])?,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OperationalIntensity {
    pub useful_stage_updates: u64,
    pub useful_signed_products: u64,
    pub compact_occurrence_reads: u64,
    pub compact_read_bytes: u64,
}

pub fn operational_intensity(
    shape: AddressMajorShape,
    tiling: StageTiling,
) -> Result<OperationalIntensity, ModelError> {
    let rows = as_u64("rows", shape.rows()?)?;
    Ok(OperationalIntensity {
        useful_stage_updates: mul("stage updates", STAGES, rows)?,
        useful_signed_products: mul("signed products", FUSED_STAGES, rows)?,
        compact_occurrence_reads: mul("occurrence reads", tiling.occurrence_passes(), rows)?,
        compact_read_bytes: mul("compact bytes", tiling.compact_bytes_per_row(), rows)?,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProductPath {
    FullWidth,
    ExactU64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceClass {
    AnalyticalControl,
    MatchedMeasurement,
}

/// Rates measured with the selected tiling and compiled accumulator shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatchedRates {
    pub evidence: EvidenceClass,
    pub product_path: ProductPath,
    pub copy_bytes_per_second: u64,
    pub base_update_lanes_per_second: u64,
    pub fused_update_lanes_per_second: u64,
    pub outer_full_products_per_second: u64,
    pub reduction_addition_lanes_per_second: u64,
}

impl MatchedRates {
    fn validate(self) -> Result<(), ModelError> {
        for (name, rate) in [
            ("copy bytes", self.copy_bytes_per_second),
            ("base updates", self.base_update_lanes_per_second),
            ("fused updates", self.fused_update_lanes_per_second),
            ("outer products", self.outer_full_products_per_second),
            (
                "reduction additions",
                self.reduction_addition_lanes_per_second,
            ),
        ] {
            if rate == 0 {
                return Err(ModelError::ZeroRate(name));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostCosts {
    pub evidence: EvidenceClass,
    pub shell_ns: u64,
    pub rounds_ns: u64,
    pub finish_and_output_ns: u64,
    pub command_boundaries: u64,
    pub command_boundary_ns: u64,
}

impl HostCosts {
    pub const ANALYTICAL_PROXY: Self = Self {
        evidence: EvidenceClass::AnalyticalControl,
        shell_ns: 0,
        rounds_ns: HOST_ROUND_PROXY_NS,
        finish_and_output_ns: 0,
        command_boundaries: 1,
        command_boundary_ns: COMMAND_BOUNDARY_NS,
    };

    pub fn total_ns(self) -> Result<u64, ModelError> {
        sum(&[
            self.shell_ns,
            self.rounds_ns,
            self.finish_and_output_ns,
            mul(
                "command boundary wall",
                self.command_boundaries,
                self.command_boundary_ns,
            )?,
        ])
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReceiptBoundRoof {
    pub product_path: ProductPath,
    pub tiling: StageTiling,
    pub outer_tiles: u64,
    pub base_update_floor_ns: u64,
    pub fused_update_floor_ns: u64,
    pub outer_product_floor_ns: u64,
    pub reduction_floor_ns: u64,
    pub compute_floor_ns: u64,
    pub traffic_floor_ns: u64,
    pub device_floor_ns: u64,
    pub device_utilization_cap_ns: u64,
    pub host_ns: u64,
    pub projected_member_ns: u64,
    pub producer_incremental_wall_ns: Option<u64>,
    pub projected_piop_charge_ns: Option<u64>,
    pub evidence_complete: bool,
    pub clears_hard_target: bool,
}

pub fn receipt_bound_roof(
    carrier: ValidatedAddressMajorCarrier,
    tiling: StageTiling,
    outer_tiles: u64,
    rates: MatchedRates,
    host: HostCosts,
    utilization_percent: u64,
) -> Result<ReceiptBoundRoof, ModelError> {
    rates.validate()?;
    if !(1..=100).contains(&utilization_percent) {
        return Err(ModelError::InvalidUtilization(utilization_percent));
    }
    let shape = carrier.shape();
    let work = work(shape, carrier.topology(), outer_tiles)?;
    let traffic = traffic(shape, carrier.topology(), tiling, outer_tiles)?;
    if traffic.member_source_row_bytes != 0 {
        return Err(ModelError::MemberSourceScan);
    }
    let base_update_floor_ns = rate_ns(
        work.issued_base_update_lanes,
        rates.base_update_lanes_per_second,
    )?;
    let fused_update_floor_ns = rate_ns(
        work.issued_fused_update_lanes,
        rates.fused_update_lanes_per_second,
    )?;
    let outer_product_floor_ns = rate_ns(
        work.issued_outer_product_lanes,
        rates.outer_full_products_per_second,
    )?;
    let reduction_floor_ns = rate_ns(
        work.issued_reduction_addition_lanes,
        rates.reduction_addition_lanes_per_second,
    )?;
    let compute_floor_ns = sum(&[
        base_update_floor_ns,
        fused_update_floor_ns,
        outer_product_floor_ns,
        reduction_floor_ns,
    ])?;
    let traffic_floor_ns = rate_ns(
        traffic.worker_forced_streaming_floor,
        rates.copy_bytes_per_second,
    )?;
    let device_floor_ns = compute_floor_ns.max(traffic_floor_ns);
    let device_utilization_cap_ns = utilization_cap(device_floor_ns, utilization_percent)?;
    let host_ns = host.total_ns()?;
    let projected_member_ns = add("projected member", host_ns, device_utilization_cap_ns)?;
    let projected_piop_charge_ns = carrier
        .producer_incremental_wall_ns()
        .map(|producer| add("projected PIOP charge", projected_member_ns, producer))
        .transpose()?;
    let evidence_complete = rates.evidence == EvidenceClass::MatchedMeasurement
        && host.evidence == EvidenceClass::MatchedMeasurement
        && carrier.producer_incremental_wall_ns().is_some();
    Ok(ReceiptBoundRoof {
        product_path: rates.product_path,
        tiling,
        outer_tiles,
        base_update_floor_ns,
        fused_update_floor_ns,
        outer_product_floor_ns,
        reduction_floor_ns,
        compute_floor_ns,
        traffic_floor_ns,
        device_floor_ns,
        device_utilization_cap_ns,
        host_ns,
        projected_member_ns,
        producer_incremental_wall_ns: carrier.producer_incremental_wall_ns(),
        projected_piop_charge_ns,
        evidence_complete,
        clears_hard_target: evidence_complete && projected_member_ns <= HARD_LOG26_TARGET_NS,
    })
}

/// Product/traffic-only screen. Missing work is explicit, so this cannot promote.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IncompleteScreen {
    pub product_path: ProductPath,
    pub product_floor_ns: u64,
    pub traffic_floor_ns: u64,
    pub partial_device_cap_ns: u64,
    pub proxy_host_ns: u64,
    pub partial_member_ns: u64,
    pub hard_target_headroom_ns: i64,
    pub missing_accumulation_lanes: u64,
    pub missing_reduction_lanes: u64,
    pub unpriced_host_equality_products: u64,
    pub complete: bool,
}

pub fn incomplete_product_screen(
    shape: AddressMajorShape,
    topology: TopologyScheduleReceipt,
    tiling: StageTiling,
    outer_tiles: u64,
    product_path: ProductPath,
) -> Result<IncompleteScreen, ModelError> {
    let work = work(shape, topology, outer_tiles)?;
    let traffic = traffic(shape, topology, tiling, outer_tiles)?;
    let outer_products = work.issued_outer_product_lanes;
    let product_floor_ns = match product_path {
        ProductPath::FullWidth => rate_ns(
            add(
                "full-width products",
                work.issued_fused_update_lanes,
                outer_products,
            )?,
            FULL_FIELD_CONSERVATIVE_PRODUCTS_PER_SECOND,
        )?,
        ProductPath::ExactU64 => add(
            "split product floors",
            rate_ns(
                work.issued_fused_update_lanes,
                SIGNED_U64_ADMISSION_TERMS_PER_SECOND,
            )?,
            rate_ns(outer_products, FULL_FIELD_CONSERVATIVE_PRODUCTS_PER_SECOND)?,
        )?,
    };
    let traffic_floor_ns = rate_ns(traffic.worker_forced_streaming_floor, COPY_BYTES_PER_SECOND)?;
    let partial_device_cap_ns = utilization_cap(product_floor_ns.max(traffic_floor_ns), 80)?;
    let proxy_host_ns = HostCosts::ANALYTICAL_PROXY.total_ns()?;
    let partial_member_ns = add("partial member", partial_device_cap_ns, proxy_host_ns)?;
    Ok(IncompleteScreen {
        product_path,
        product_floor_ns,
        traffic_floor_ns,
        partial_device_cap_ns,
        proxy_host_ns,
        partial_member_ns,
        hard_target_headroom_ns: signed_difference(HARD_LOG26_TARGET_NS, partial_member_ns)?,
        missing_accumulation_lanes: add(
            "missing accumulation lanes",
            work.issued_base_update_lanes,
            work.issued_fused_update_lanes,
        )?,
        missing_reduction_lanes: work.issued_reduction_addition_lanes,
        unpriced_host_equality_products: work.host_equality_generation_products,
        complete: false,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelError {
    Carrier(CarrierError),
    InvalidTopology,
    InvalidOuterTiles { tiles: u64, outer_length: u64 },
    InvalidUtilization(u64),
    MemberSourceScan,
    ZeroRate(&'static str),
    Overflow(&'static str),
}

impl From<CarrierError> for ModelError {
    fn from(value: CarrierError) -> Self {
        Self::Carrier(value)
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Carrier(error) => error.fmt(f),
            Self::InvalidTopology => f.write_str("invalid measured topology"),
            Self::InvalidOuterTiles {
                tiles,
                outer_length,
            } => write!(
                f,
                "invalid outer tile count {tiles} for {outer_length} outer blocks"
            ),
            Self::InvalidUtilization(value) => write!(f, "invalid utilization {value}%"),
            Self::MemberSourceScan => f.write_str("receipt admitted a member source-row scan"),
            Self::ZeroRate(name) => write!(f, "{name} rate is zero"),
            Self::Overflow(name) => write!(f, "{name} overflowed"),
        }
    }
}

impl std::error::Error for ModelError {}

fn validate_outer_tiles(outer_length: u64, tiles: u64) -> Result<(), ModelError> {
    if tiles == 0 || tiles > outer_length {
        Err(ModelError::InvalidOuterTiles {
            tiles,
            outer_length,
        })
    } else {
        Ok(())
    }
}

fn as_u64(name: &'static str, value: usize) -> Result<u64, ModelError> {
    u64::try_from(value).map_err(|_| ModelError::Overflow(name))
}

fn add(name: &'static str, left: u64, right: u64) -> Result<u64, ModelError> {
    left.checked_add(right).ok_or(ModelError::Overflow(name))
}

fn mul(name: &'static str, left: u64, right: u64) -> Result<u64, ModelError> {
    left.checked_mul(right).ok_or(ModelError::Overflow(name))
}

fn sum(values: &[u64]) -> Result<u64, ModelError> {
    values
        .iter()
        .try_fold(0, |total, value| add("sum", total, *value))
}

fn round_up(name: &'static str, value: u64, multiple: u64) -> Result<u64, ModelError> {
    if multiple == 0 {
        return Err(ModelError::ZeroRate(name));
    }
    value
        .div_ceil(multiple)
        .checked_mul(multiple)
        .ok_or(ModelError::Overflow(name))
}

fn rate_ns(units: u64, units_per_second: u64) -> Result<u64, ModelError> {
    if units_per_second == 0 {
        return Err(ModelError::ZeroRate("roof"));
    }
    let numerator = u128::from(units)
        .checked_mul(1_000_000_000)
        .ok_or(ModelError::Overflow("rate numerator"))?;
    u64::try_from(numerator.div_ceil(u128::from(units_per_second)))
        .map_err(|_| ModelError::Overflow("rate time"))
}

fn utilization_cap(floor_ns: u64, utilization_percent: u64) -> Result<u64, ModelError> {
    if !(1..=100).contains(&utilization_percent) {
        return Err(ModelError::InvalidUtilization(utilization_percent));
    }
    let numerator = u128::from(floor_ns)
        .checked_mul(100)
        .ok_or(ModelError::Overflow("utilization numerator"))?;
    u64::try_from(numerator.div_ceil(u128::from(utilization_percent)))
        .map_err(|_| ModelError::Overflow("utilization cap"))
}

fn signed_difference(left: u64, right: u64) -> Result<i64, ModelError> {
    let difference = i128::from(left) - i128::from(right);
    i64::try_from(difference).map_err(|_| ModelError::Overflow("signed difference"))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests use checked fixtures")]
mod tests {
    use super::super::carrier::{
        CountPublication, PlaneReceipt, ProducerIdentity, ScatterPublication,
        ValidatedProducerCounts, CELL_BYTES, INNER_SIGN_BYTES, MAGNITUDE_BYTES, RESIDENT_ROW_BYTES,
    };
    use super::*;

    fn ready_carrier() -> ValidatedAddressMajorCarrier {
        let shape = AddressMajorShape::LOG26;
        let rows = shape.rows().unwrap();
        let cells_count = shape.cells().unwrap();
        let producer = ProducerIdentity::new(5, 7, rows * RESIDENT_ROW_BYTES, 11, rows).unwrap();
        let cells = PlaneReceipt::new(13, cells_count, cells_count * CELL_BYTES).unwrap();
        let counts = ValidatedProducerCounts::publish(
            shape,
            producer,
            cells,
            CountPublication {
                initialized_cells: cells_count,
                count_updates: rows,
                counted_rows: rows,
                completed_outer_blocks: shape.outer_length().unwrap(),
                invalid_rows: 0,
                reserved: [0; 3],
                additional_source_scans: 0,
                member_source_read_bytes: 0,
                host_staging_bytes: 0,
                upload_copy_bytes: 0,
            },
        )
        .unwrap();
        let topology = MeasuredTopology::FIBONACCI_LOG26
            .schedule_bounds(shape)
            .unwrap()
            .pessimistic;
        ValidatedAddressMajorCarrier::publish(
            counts,
            PlaneReceipt::new(17, rows, rows * INNER_SIGN_BYTES).unwrap(),
            PlaneReceipt::new(19, rows, rows * MAGNITUDE_BYTES).unwrap(),
            topology,
            ScatterPublication {
                scattered_rows: rows,
                cursor_updates: rows,
                completed_outer_blocks: shape.outer_length().unwrap(),
                invalid_rows: 0,
                reserved: [0; 3],
                producer_resident_scans: 1,
                member_resident_scans: 0,
                source_requested_bytes: 16 * rows as u64,
                compact_write_bytes: 12 * rows as u64,
                cell_write_bytes: 4 * cells_count as u64,
                member_source_read_bytes: 0,
                host_staging_bytes: 0,
                upload_copy_bytes: 0,
                first_push_pc: 0,
                producer_incremental_wall_ns: Some(4_500_000),
                producer_gpu_active_ns: Some(4_000_000),
            },
        )
        .unwrap()
    }

    #[test]
    fn hard_target_is_stricter_than_the_derived_family_allowance() {
        assert_eq!(STANDALONE_FIVE_X_CAP_NS, 40_817_025);
        assert_eq!(DERIVED_FAMILY_SEVEN_X_ADDRESS_NS, 27_735_322);
        assert_eq!(HARD_LOG26_TARGET_NS, 27_700_000);
        const {
            assert!(HARD_LOG26_TARGET_NS < DERIVED_FAMILY_SEVEN_X_ADDRESS_NS);
        }
    }

    #[test]
    fn measured_schedule_bounds_are_exact() {
        let bounds = MeasuredTopology::FIBONACCI_LOG26
            .schedule_bounds(AddressMajorShape::LOG26)
            .unwrap();
        assert_eq!(bounds.optimistic.padded_short_lanes, 33_888);
        assert_eq!(bounds.optimistic.padded_long_lanes, 67_107_648);
        assert_eq!(bounds.pessimistic.padded_short_lanes, 33_888);
        assert_eq!(bounds.pessimistic.padded_long_lanes, 67_695_040);
    }

    #[test]
    fn pessimistic_work_freezes_the_real_census_screen() {
        let topology = MeasuredTopology::FIBONACCI_LOG26
            .schedule_bounds(AddressMajorShape::LOG26)
            .unwrap()
            .pessimistic;
        let work = work(AddressMajorShape::LOG26, topology, 8).unwrap();
        assert_eq!(work.useful_fused_updates, 268_435_456);
        assert_eq!(work.useful_outer_products, 180_072);
        assert_eq!(work.issued_fused_update_lanes, 270_915_712);
        assert_eq!(work.issued_outer_product_lanes, 5_762_304);
        assert_eq!(work.host_equality_generation_products, 626_652);
        assert_eq!(
            work.issued_base_update_lanes + work.issued_fused_update_lanes,
            609_560_352
        );
        assert_eq!(work.simd_run_reduction_lanes, 28_811_520);
        assert_eq!(work.threadgroup_reduction_lanes, 33_554_432);
        assert_eq!(work.tile_reduction_lanes, 589_824);
        assert_eq!(work.issued_reduction_addition_lanes, 62_955_776);
        assert_eq!(work.producer_count_cursor_atomics, 134_217_728);
        assert_eq!(work.member_output_atomics, 0);
    }

    #[test]
    fn traffic_prices_the_second_compact_pass() {
        let topology = MeasuredTopology::FIBONACCI_LOG26
            .schedule_bounds(AddressMajorShape::LOG26)
            .unwrap()
            .pessimistic;
        let split = traffic(
            AddressMajorShape::LOG26,
            topology,
            StageTiling::FiveThenFour,
            8,
        )
        .unwrap();
        let single = traffic(
            AddressMajorShape::LOG26,
            topology,
            StageTiling::NineSinglePass,
            8,
        )
        .unwrap();
        assert_eq!(split.producer_cursor_requested, 1_946_157_056);
        assert_eq!(split.producer_rank_scatter_requested, 2_080_374_784);
        assert_eq!(split.producer_rank_whole_piop_requested, 2_214_592_512);
        assert_eq!(split.member_source_row_bytes, 0);
        assert_eq!(split.packed_cell_read_bytes, 134_217_728);
        assert_eq!(split.partial_write_read_bytes, 18_874_368);
        assert_eq!(split.worker_physical_unique_minimum, 888_045_568);
        assert_eq!(split.worker_forced_streaming_floor, 1_233_027_072);
        assert_eq!(split.worker_shader_requested, 10_894_571_136);
        assert_eq!(single.worker_forced_streaming_floor, 897_482_752);
        assert_eq!(single.worker_shader_requested, 10_559_026_816);
        assert_eq!(
            split.compact_read_bytes - single.compact_read_bytes,
            268_435_456
        );
    }

    #[test]
    fn incomplete_screens_cannot_spend_unmeasured_headroom() {
        let topology = MeasuredTopology::FIBONACCI_LOG26
            .schedule_bounds(AddressMajorShape::LOG26)
            .unwrap()
            .pessimistic;
        let full = incomplete_product_screen(
            AddressMajorShape::LOG26,
            topology,
            StageTiling::FiveThenFour,
            8,
            ProductPath::FullWidth,
        )
        .unwrap();
        let exact = incomplete_product_screen(
            AddressMajorShape::LOG26,
            topology,
            StageTiling::FiveThenFour,
            8,
            ProductPath::ExactU64,
        )
        .unwrap();
        assert_eq!(full.product_floor_ns, 15_286_079);
        assert_eq!(full.traffic_floor_ns, 2_729_738);
        assert_eq!(full.partial_member_ns, 27_166_850);
        assert_eq!(full.hard_target_headroom_ns, 533_150);
        assert_eq!(exact.product_floor_ns, 10_630_317);
        assert_eq!(exact.partial_member_ns, 21_347_148);
        assert_eq!(exact.hard_target_headroom_ns, 6_352_852);
        assert!(!full.complete);
        assert!(!exact.complete);
        assert_eq!(full.missing_accumulation_lanes, 609_560_352);
        assert_eq!(full.missing_reduction_lanes, 62_955_776);
        assert_eq!(full.unpriced_host_equality_products, 626_652);
    }

    #[test]
    fn only_matched_receipts_can_clear_the_hard_target() {
        let carrier = ready_carrier();
        let rates = MatchedRates {
            evidence: EvidenceClass::MatchedMeasurement,
            product_path: ProductPath::ExactU64,
            copy_bytes_per_second: COPY_BYTES_PER_SECOND,
            base_update_lanes_per_second: 500_000_000_000,
            fused_update_lanes_per_second: 500_000_000_000,
            outer_full_products_per_second: 50_000_000_000,
            reduction_addition_lanes_per_second: 500_000_000_000,
        };
        let host = HostCosts {
            evidence: EvidenceClass::MatchedMeasurement,
            shell_ns: 500_000,
            rounds_ns: HOST_ROUND_PROXY_NS,
            finish_and_output_ns: 200_000,
            command_boundaries: 1,
            command_boundary_ns: COMMAND_BOUNDARY_NS,
        };
        let complete =
            receipt_bound_roof(carrier, StageTiling::FiveThenFour, 8, rates, host, 80).unwrap();
        assert!(complete.evidence_complete);
        assert!(complete.clears_hard_target);
        assert_eq!(
            complete.projected_piop_charge_ns,
            Some(complete.projected_member_ns + 4_500_000)
        );

        let analytical = receipt_bound_roof(
            carrier,
            StageTiling::FiveThenFour,
            8,
            MatchedRates {
                evidence: EvidenceClass::AnalyticalControl,
                ..rates
            },
            host,
            80,
        )
        .unwrap();
        assert!(!analytical.evidence_complete);
        assert!(!analytical.clears_hard_target);
    }
}
