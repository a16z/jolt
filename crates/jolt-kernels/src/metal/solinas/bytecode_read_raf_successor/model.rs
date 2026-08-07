//! Exact integer model for the address-major bytecode read/RAF design.
//!
//! This file is intentionally unregistered. A projection is a necessary
//! screen until matched addition, reduction, and topology-atomic rates exist.

use core::fmt;

pub const LOG26_ROWS: u64 = 1 << 26;
pub const ADDRESS_LOG2: u32 = 13;
pub const INNER_LOG2: u32 = 15;
pub const STAGES: u64 = 9;
pub const BASE_STAGES: u64 = 5;
pub const FUSED_STAGES: u64 = 4;
pub const SIMD_WIDTH: u64 = 32;
pub const SHORT_THRESHOLD: u64 = 32;

pub const CPU_LOG26_SAMPLES_NS: [u64; 5] = [
    172_796_544,
    198_165_708,
    181_211_502,
    190_915_958,
    198_945_292,
];
pub const CPU_LOG26_MEDIAN_NS: u64 = 190_915_958;
pub const CPU_PREPARE_MEDIAN_NS: u64 = 182_930_333;
pub const CPU_HOST_ROUNDS_TOTAL_NS: u64 = 7_918_251;
pub const FIVE_X_CAP_NS: u64 = CPU_LOG26_MEDIAN_NS / 5;
pub const EIGHT_X_CAP_NS: u64 = CPU_LOG26_MEDIAN_NS / 8;

pub const CYCLE_CPU_LOG26_MEDIAN_NS: u64 = 1_004_692_916;
pub const CYCLE_METAL_LOG26_MEDIAN_NS: u64 = 160_876_418;
pub const PAIRED_ADDRESS_CYCLE_CPU_MEDIAN_NS: u64 = 1_203_638_208;
pub const PAIRED_FIVE_X_CAP_NS: u64 = PAIRED_ADDRESS_CYCLE_CPU_MEDIAN_NS / 5;
pub const PAIRED_EIGHT_X_CAP_NS: u64 = PAIRED_ADDRESS_CYCLE_CPU_MEDIAN_NS / 8;

pub const COPY_BYTES_PER_SECOND: u64 = 451_701_710_520;
pub const SIGNED_U64_PEAK_TERMS_PER_SECOND: u64 = 70_417_000_000;
pub const FULL_FIELD_PEAK_PRODUCTS_PER_SECOND: u64 = 45_709_000_000;
pub const SIGNED_U64_ADMISSION_TERMS_PER_SECOND: u64 = 26_272_000_000;
pub const FULL_FIELD_CONSERVATIVE_PRODUCTS_PER_SECOND: u64 = 18_100_000_000;
pub const COMMAND_BOUNDARY_NS: u64 = 141_000;

const FIELD_BYTES: u64 = 16;
const LAYOUT_BYTES: u64 = 4;
const INNER_SIGN_BYTES: u64 = 4;
const MAGNITUDE_BYTES: u64 = 8;
const RESIDENT_ROW_BYTES: u64 = 40;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub log_rows: u32,
    pub log_addresses: u32,
    pub inner_log2: u32,
}

impl Shape {
    pub const LOG26: Self = Self {
        log_rows: 26,
        log_addresses: ADDRESS_LOG2,
        inner_log2: INNER_LOG2,
    };

    pub const LOG28: Self = Self {
        log_rows: 28,
        log_addresses: ADDRESS_LOG2,
        inner_log2: INNER_LOG2,
    };

    pub fn validate(self) -> Result<(), ModelError> {
        if self.log_rows < self.inner_log2
            || self.log_rows >= 63
            || self.log_addresses >= 63
            || self.inner_log2 == 0
        {
            return Err(ModelError::InvalidShape);
        }
        Ok(())
    }

    pub fn rows(self) -> Result<u64, ModelError> {
        self.validate()?;
        Ok(1_u64 << self.log_rows)
    }

    pub fn addresses(self) -> Result<u64, ModelError> {
        self.validate()?;
        Ok(1_u64 << self.log_addresses)
    }

    pub fn inner_length(self) -> Result<u64, ModelError> {
        self.validate()?;
        Ok(1_u64 << self.inner_log2)
    }

    pub fn outer_length(self) -> Result<u64, ModelError> {
        Ok(self.rows()? / self.inner_length()?)
    }

    pub fn cells(self) -> Result<u64, ModelError> {
        mul(self.addresses()?, self.outer_length()?)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryFootprint {
    pub shared_rows: u64,
    pub layout: u64,
    pub inner_sign: u64,
    pub magnitude: u64,
    pub equality_lo: u64,
    pub equality_hi: u64,
    pub pushforwards: u64,
    pub successor_owned: u64,
    pub aggregate_with_shared_rows: u64,
}

pub fn memory_footprint(shape: Shape) -> Result<MemoryFootprint, ModelError> {
    let rows = shape.rows()?;
    let layout = mul(LAYOUT_BYTES, shape.cells()?)?;
    let inner_sign = mul(INNER_SIGN_BYTES, rows)?;
    let magnitude = mul(MAGNITUDE_BYTES, rows)?;
    let equality_lo = mul(mul(FIELD_BYTES, STAGES)?, shape.inner_length()?)?;
    let equality_hi = mul(mul(FIELD_BYTES, STAGES)?, shape.outer_length()?)?;
    let pushforwards = mul(mul(FIELD_BYTES, STAGES)?, shape.addresses()?)?;
    let successor_owned = sum(&[
        layout,
        inner_sign,
        magnitude,
        equality_lo,
        equality_hi,
        pushforwards,
    ])?;
    let shared_rows = mul(RESIDENT_ROW_BYTES, rows)?;
    Ok(MemoryFootprint {
        shared_rows,
        layout,
        inner_sign,
        magnitude,
        equality_lo,
        equality_hi,
        pushforwards,
        successor_owned,
        aggregate_with_shared_rows: add(successor_owned, shared_rows)?,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TopologyStats {
    pub short_occurrences: u64,
    pub long_occurrences: u64,
    pub short_runs: u64,
    pub long_runs: u64,
    pub short_batches: u64,
    pub padded_short_lanes: u64,
    pub padded_long_lanes: u64,
}

impl TopologyStats {
    pub fn validate(self, shape: Shape) -> Result<(), ModelError> {
        let rows = shape.rows()?;
        let runs = add(self.short_runs, self.long_runs)?;
        if add(self.short_occurrences, self.long_occurrences)? != rows
            || runs < shape.outer_length()?
            || runs > shape.cells()?
            || self.short_occurrences < self.short_runs
            || self.short_occurrences > mul(SHORT_THRESHOLD, self.short_runs)?
            || self.long_occurrences < mul(SHORT_THRESHOLD + 1, self.long_runs)?
            || self.long_occurrences > mul(shape.inner_length()?, self.long_runs)?
        {
            return Err(ModelError::InvalidTopology);
        }

        if self.short_runs == 0 {
            if self.short_occurrences != 0
                || self.short_batches != 0
                || self.padded_short_lanes != 0
            {
                return Err(ModelError::InvalidTopology);
            }
        } else if self.short_batches == 0
            || self.short_batches > self.short_runs
            || self.short_runs > mul(SIMD_WIDTH, self.short_batches)?
            || self.padded_short_lanes < self.short_occurrences
            || self.padded_short_lanes > mul(mul(SIMD_WIDTH, SHORT_THRESHOLD)?, self.short_batches)?
            || self.padded_short_lanes % SIMD_WIDTH != 0
        {
            return Err(ModelError::InvalidTopology);
        }

        if self.long_runs == 0 {
            if self.long_occurrences != 0 || self.padded_long_lanes != 0 {
                return Err(ModelError::InvalidTopology);
            }
        } else if self.padded_long_lanes < self.long_occurrences
            || self.padded_long_lanes
                >= add(self.long_occurrences, mul(SIMD_WIDTH - 1, self.long_runs)?)?
            || self.padded_long_lanes % SIMD_WIDTH != 0
        {
            return Err(ModelError::InvalidTopology);
        }
        Ok(())
    }

    pub fn runs(self) -> Result<u64, ModelError> {
        add(self.short_runs, self.long_runs)
    }

    pub fn padded_lanes(self) -> Result<u64, ModelError> {
        add(self.padded_short_lanes, self.padded_long_lanes)
    }

    pub fn minimum_runs(shape: Shape) -> Result<Self, ModelError> {
        let rows = shape.rows()?;
        let runs = shape.outer_length()?;
        let stats = Self {
            short_occurrences: 0,
            long_occurrences: rows,
            short_runs: 0,
            long_runs: runs,
            short_batches: 0,
            padded_short_lanes: 0,
            padded_long_lanes: rows,
        };
        stats.validate(shape)?;
        Ok(stats)
    }

    pub fn dense_four_per_cell(shape: Shape) -> Result<Self, ModelError> {
        if shape.inner_length()? != mul(4, shape.addresses()?)?
            || shape.outer_length()? % SIMD_WIDTH != 0
        {
            return Err(ModelError::InvalidTopology);
        }
        let rows = shape.rows()?;
        let runs = shape.cells()?;
        let short_batches = mul(shape.addresses()?, shape.outer_length()? / SIMD_WIDTH)?;
        let stats = Self {
            short_occurrences: rows,
            long_occurrences: 0,
            short_runs: runs,
            long_runs: 0,
            short_batches,
            padded_short_lanes: rows,
            padded_long_lanes: 0,
        };
        stats.validate(shape)?;
        Ok(stats)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Work {
    pub useful_signed_u64_products: u64,
    pub useful_full_products: u64,
    pub issued_signed_u64_product_lanes: u64,
    pub issued_full_product_lanes: u64,
    pub issued_accumulation_addition_lanes: u64,
    pub issued_reduction_addition_lanes: u64,
    pub topology_threadgroup_atomics: u64,
    pub equality_generation_full_products: u64,
}

pub fn work(shape: Shape, topology: TopologyStats) -> Result<Work, ModelError> {
    topology.validate(shape)?;
    let rows = shape.rows()?;
    let padded = topology.padded_lanes()?;
    let runs = topology.runs()?;
    let short_full = mul(mul(STAGES, SIMD_WIDTH)?, topology.short_batches)?;
    let long_full = mul(SIMD_WIDTH, topology.long_runs)?;
    let reductions = mul(1_440, add(topology.short_batches, topology.long_runs)?)?;
    let final_reductions = mul(mul(SIMD_WIDTH, 8)?, shape.addresses()?)?;
    Ok(Work {
        useful_signed_u64_products: mul(FUSED_STAGES, rows)?,
        useful_full_products: mul(STAGES, runs)?,
        issued_signed_u64_product_lanes: mul(FUSED_STAGES, padded)?,
        issued_full_product_lanes: add(short_full, long_full)?,
        issued_accumulation_addition_lanes: mul(STAGES, padded)?,
        issued_reduction_addition_lanes: add(reductions, final_reductions)?,
        topology_threadgroup_atomics: mul(2, rows)?,
        equality_generation_full_products: equality_generation_products(shape)?,
    })
}

pub fn equality_generation_products(shape: Shape) -> Result<u64, ModelError> {
    let table_nodes = add(
        shape
            .inner_length()?
            .checked_sub(1)
            .ok_or(ModelError::Overflow)?,
        shape
            .outer_length()?
            .checked_sub(1)
            .ok_or(ModelError::Overflow)?,
    )?;
    mul(mul(2, STAGES)?, table_nodes)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Traffic {
    pub topology_requested: u64,
    pub topology_unique_minimum: u64,
    pub topology_uncached_row_upper: u64,
    pub worker_physical_unique_minimum: u64,
    pub worker_shader_requested: u64,
}

pub fn traffic(shape: Shape, topology: TopologyStats) -> Result<Traffic, ModelError> {
    topology.validate(shape)?;
    let rows = shape.rows()?;
    let layout = mul(LAYOUT_BYTES, shape.cells()?)?;
    let memory = memory_footprint(shape)?;
    let worker_shader_requested = sum(&[
        mul(160, rows)?,
        layout,
        mul(mul(FIELD_BYTES, STAGES)?, topology.runs()?)?,
        memory.pushforwards,
    ])?;
    Ok(Traffic {
        topology_requested: add(mul(36, rows)?, layout)?,
        topology_unique_minimum: add(mul(28, rows)?, layout)?,
        topology_uncached_row_upper: add(mul(92, rows)?, layout)?,
        worker_physical_unique_minimum: memory.successor_owned,
        worker_shader_requested,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArithmeticRates {
    IsolatedPeak,
    AdmissionFloor,
}

impl ArithmeticRates {
    const fn signed_rate(self) -> u64 {
        match self {
            Self::IsolatedPeak => SIGNED_U64_PEAK_TERMS_PER_SECOND,
            Self::AdmissionFloor => SIGNED_U64_ADMISSION_TERMS_PER_SECOND,
        }
    }

    const fn full_rate(self) -> u64 {
        match self {
            Self::IsolatedPeak => FULL_FIELD_PEAK_PRODUCTS_PER_SECOND,
            Self::AdmissionFloor => FULL_FIELD_CONSERVATIVE_PRODUCTS_PER_SECOND,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TopologyCharge {
    ResidentProducer,
    ChargeToMember,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NecessaryScreen {
    pub arithmetic_rates: ArithmeticRates,
    pub topology_charge: TopologyCharge,
    pub product_floor_ns: u64,
    pub worker_traffic_floor_ns: u64,
    pub worker_active_floor_ns: u64,
    pub worker_eighty_percent_cap_ns: u64,
    pub topology_traffic_floor_ns: u64,
    pub topology_eighty_percent_cap_ns: u64,
    pub host_and_command_ns: u64,
    pub projected_member_ns: u64,
    pub projected_speedup_milli_x: u64,
    pub complete: bool,
}

pub fn necessary_screen(
    shape: Shape,
    topology: TopologyStats,
    arithmetic_rates: ArithmeticRates,
    topology_charge: TopologyCharge,
) -> Result<NecessaryScreen, ModelError> {
    let work = work(shape, topology)?;
    let traffic = traffic(shape, topology)?;
    let product_floor_ns = add(
        rate_ns(
            work.issued_signed_u64_product_lanes,
            arithmetic_rates.signed_rate(),
        )?,
        rate_ns(
            add(
                work.issued_full_product_lanes,
                work.equality_generation_full_products,
            )?,
            arithmetic_rates.full_rate(),
        )?,
    )?;
    let worker_traffic_floor_ns = rate_ns(
        traffic.worker_physical_unique_minimum,
        COPY_BYTES_PER_SECOND,
    )?;
    let worker_active_floor_ns = product_floor_ns.max(worker_traffic_floor_ns);
    let worker_eighty_percent_cap_ns = eighty_percent_cap(worker_active_floor_ns)?;
    let topology_traffic_floor_ns = match topology_charge {
        TopologyCharge::ResidentProducer => 0,
        TopologyCharge::ChargeToMember => {
            rate_ns(traffic.topology_requested, COPY_BYTES_PER_SECOND)?
        }
    };
    let topology_eighty_percent_cap_ns = eighty_percent_cap(topology_traffic_floor_ns)?;
    let command_count = match topology_charge {
        TopologyCharge::ResidentProducer => 1,
        TopologyCharge::ChargeToMember => 2,
    };
    let host_and_command_ns = add(
        CPU_HOST_ROUNDS_TOTAL_NS,
        mul(COMMAND_BOUNDARY_NS, command_count)?,
    )?;
    let projected_member_ns = sum(&[
        host_and_command_ns,
        worker_eighty_percent_cap_ns,
        topology_eighty_percent_cap_ns,
    ])?;
    Ok(NecessaryScreen {
        arithmetic_rates,
        topology_charge,
        product_floor_ns,
        worker_traffic_floor_ns,
        worker_active_floor_ns,
        worker_eighty_percent_cap_ns,
        topology_traffic_floor_ns,
        topology_eighty_percent_cap_ns,
        host_and_command_ns,
        projected_member_ns,
        projected_speedup_milli_x: mul(CPU_LOG26_MEDIAN_NS, 1_000)? / projected_member_ns,
        complete: false,
    })
}

pub fn estimated_cpu_member_ns(rows: u64) -> Result<u64, ModelError> {
    add(
        mul(CPU_PREPARE_MEDIAN_NS, rows)? / LOG26_ROWS,
        CPU_HOST_ROUNDS_TOTAL_NS,
    )
}

fn rate_ns(units: u64, units_per_second: u64) -> Result<u64, ModelError> {
    if units_per_second == 0 {
        return Err(ModelError::ZeroRate);
    }
    let numerator = u128::from(units)
        .checked_mul(1_000_000_000)
        .ok_or(ModelError::Overflow)?;
    let denominator = u128::from(units_per_second);
    u64::try_from(numerator.div_ceil(denominator)).map_err(|_| ModelError::Overflow)
}

fn eighty_percent_cap(floor_ns: u64) -> Result<u64, ModelError> {
    let value = u128::from(floor_ns)
        .checked_mul(5)
        .ok_or(ModelError::Overflow)?
        .div_ceil(4);
    u64::try_from(value).map_err(|_| ModelError::Overflow)
}

fn sum(values: &[u64]) -> Result<u64, ModelError> {
    values.iter().try_fold(0, |acc, value| add(acc, *value))
}

fn add(left: u64, right: u64) -> Result<u64, ModelError> {
    left.checked_add(right).ok_or(ModelError::Overflow)
}

fn mul(left: u64, right: u64) -> Result<u64, ModelError> {
    left.checked_mul(right).ok_or(ModelError::Overflow)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelError {
    InvalidShape,
    InvalidTopology,
    ZeroRate,
    Overflow,
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => f.write_str("invalid bytecode successor shape"),
            Self::InvalidTopology => f.write_str("invalid bytecode successor topology census"),
            Self::ZeroRate => f.write_str("bytecode successor roof rate is zero"),
            Self::Overflow => f.write_str("bytecode successor model arithmetic overflowed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_caps_use_strict_integer_division() {
        assert_eq!(FIVE_X_CAP_NS, 38_183_191);
        assert_eq!(EIGHT_X_CAP_NS, 23_864_494);
        assert_eq!(PAIRED_FIVE_X_CAP_NS, 240_727_641);
        assert_eq!(PAIRED_EIGHT_X_CAP_NS, 150_454_776);
    }

    #[test]
    fn log26_memory_is_exact() {
        let memory = memory_footprint(Shape::LOG26).unwrap();
        assert_eq!(memory.layout, 67_108_864);
        assert_eq!(memory.inner_sign, 268_435_456);
        assert_eq!(memory.magnitude, 536_870_912);
        assert_eq!(memory.equality_lo, 4_718_592);
        assert_eq!(memory.equality_hi, 294_912);
        assert_eq!(memory.pushforwards, 1_179_648);
        assert_eq!(memory.successor_owned, 878_608_384);
        assert_eq!(memory.aggregate_with_shared_rows, 3_562_962_944);
    }

    #[test]
    fn log28_capacity_is_exact() {
        let memory = memory_footprint(Shape::LOG28).unwrap();
        assert_eq!(memory.layout, 268_435_456);
        assert_eq!(memory.successor_owned, 3_496_738_816);
        assert_eq!(memory.aggregate_with_shared_rows, 14_234_157_056);
    }

    #[test]
    fn topology_extremes_have_expected_issue_counts() {
        let minimum = work(
            Shape::LOG26,
            TopologyStats::minimum_runs(Shape::LOG26).unwrap(),
        )
        .unwrap();
        assert_eq!(minimum.useful_signed_u64_products, 268_435_456);
        assert_eq!(minimum.useful_full_products, 18_432);
        assert_eq!(minimum.issued_full_product_lanes, 65_536);

        let dense = work(
            Shape::LOG26,
            TopologyStats::dense_four_per_cell(Shape::LOG26).unwrap(),
        )
        .unwrap();
        assert_eq!(dense.useful_full_products, 150_994_944);
        assert_eq!(dense.issued_full_product_lanes, 150_994_944);
        assert_eq!(dense.issued_reduction_addition_lanes, 757_071_872);
    }

    #[test]
    fn traffic_views_do_not_mix_requested_and_physical_bytes() {
        let minimum = TopologyStats::minimum_runs(Shape::LOG26).unwrap();
        let traffic = traffic(Shape::LOG26, minimum).unwrap();
        assert_eq!(traffic.topology_requested, 2_483_027_968);
        assert_eq!(traffic.topology_unique_minimum, 1_946_157_056);
        assert_eq!(traffic.topology_uncached_row_upper, 6_241_124_352);
        assert_eq!(traffic.worker_physical_unique_minimum, 878_608_384);
        assert_eq!(traffic.worker_shader_requested, 10_806_001_664);
    }

    #[test]
    fn screens_remain_marked_incomplete_without_matched_addition_rates() {
        let dense = TopologyStats::dense_four_per_cell(Shape::LOG26).unwrap();
        let peak = necessary_screen(
            Shape::LOG26,
            dense,
            ArithmeticRates::IsolatedPeak,
            TopologyCharge::ChargeToMember,
        )
        .unwrap();
        let admission = necessary_screen(
            Shape::LOG26,
            dense,
            ArithmeticRates::AdmissionFloor,
            TopologyCharge::ChargeToMember,
        )
        .unwrap();
        assert!(!peak.complete);
        assert!(!admission.complete);
        assert!(peak.projected_member_ns < EIGHT_X_CAP_NS + 200_000);
        assert!(admission.projected_member_ns > FIVE_X_CAP_NS);
    }
}
