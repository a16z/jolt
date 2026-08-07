//! Exact traffic, latency, and campaign controls for the retained-hot path.

use core::cmp::Ordering;

use super::{
    HammingWeightV2Error, HammingWeightV2Geometry, HAMMING_V2_BINS, HAMMING_V2_FIELD_BYTES,
    HAMMING_V2_HOT_PLANES, HAMMING_V2_ROW_BYTES, HAMMING_V2_SELECTORS, HAMMING_V2_VALIDITY_PLANES,
};

pub const M4_MAX_COPY_BYTES_PER_SECOND: u64 = 451_701_710_520;
pub const DEVICE_MAX_BUFFER_BYTES: u64 = 86_586_540_032;

/// Fresh exact log-26 diagnostic which explicitly selected `accepted-rows`.
pub const CURRENT_ACCEPTED_CPU_MEMBER_NS: u64 = 549_294_665;
pub const CURRENT_ACCEPTED_METAL_MEMBER_NS: u64 = 111_356_377;
pub const CURRENT_ACCEPTED_GPU_ACTIVE_NS: u64 = 86_162_042;
pub const CURRENT_ACCEPTED_NON_GPU_NS: u64 =
    CURRENT_ACCEPTED_METAL_MEMBER_NS - CURRENT_ACCEPTED_GPU_ACTIVE_NS;

/// Lower observed speedups from the existing retained implementation.
/// These are evidence routing constants, never an acceptance substitute.
pub const RETAINED_LOG_26_MIN_SPEEDUP_MILLI_X: u64 = 7_863;
pub const RETAINED_LOG_27_DIAGNOSTIC_SPEEDUP_MILLI_X: u64 = 7_218;
pub const FROZEN_SELECTOR_ROW_OPPORTUNITIES: u64 = 1_946_157_056;
pub const FROZEN_RETAINED_NONZERO_ADDS: u64 = 1_588_505_707;
pub const ROBUST_ATOMIC_SERVICE_CONTROL_NS: u64 = 35_451_625;

pub const HARD_FLOOR_NUMERATOR: u64 = 5;
pub const HARD_FLOOR_DENOMINATOR: u64 = 1;
pub const ROBUST_BAR_NUMERATOR: u64 = 53;
pub const ROBUST_BAR_DENOMINATOR: u64 = 10;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HammingWeightV2Candidate {
    AcceptedRowsControl,
    ExistingRetainedHot,
}

/// The accepted-row miss is not a shader-design result: the existing retained
/// path already clears the target at both measured scales.
pub const fn next_candidate() -> HammingWeightV2Candidate {
    HammingWeightV2Candidate::ExistingRetainedHot
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingWeightTrafficModel {
    pub rows: u64,
    pub accepted_row_scan_bytes: u64,
    pub projection_write_bytes: u64,
    pub retained_hot_bytes: u64,
    pub validity_bytes: u64,
    pub retained_consumer_read_bytes: u64,
    pub equality_cache_unique_bytes: u64,
    pub equality_fully_issued_bytes: u64,
    pub partial_write_read_bytes: u64,
    pub output_write_read_bytes: u64,
    pub accepted_cache_optimistic_bytes: u64,
    pub retained_cache_optimistic_bytes: u64,
    pub accepted_fully_issued_bytes: u64,
    pub retained_fully_issued_bytes: u64,
    pub fused_producer_plus_consumer_bytes: u64,
    pub consumer_owned_bytes: u64,
    pub retained_copy_floor_ns: u64,
    pub retained_eighty_percent_copy_cap_ns: u64,
}

impl HammingWeightTrafficModel {
    pub fn new(geometry: HammingWeightV2Geometry) -> Result<Self, HammingWeightV2Error> {
        let rows = geometry.rows() as u64;
        let e_in = geometry.e_in_length() as u64;
        let e_out = geometry.e_out_length() as u64;
        let tiles = 5u64;
        let accepted_row_scan_bytes = product(&[tiles, HAMMING_V2_ROW_BYTES, rows])?;
        let projection_write_bytes = product(&[
            HAMMING_V2_HOT_PLANES as u64 + HAMMING_V2_VALIDITY_PLANES as u64,
            rows,
        ])?;
        let retained_hot_bytes = product(&[HAMMING_V2_HOT_PLANES as u64, rows])?;
        let validity_bytes = product(&[HAMMING_V2_VALIDITY_PLANES as u64, rows])?;
        let retained_consumer_read_bytes = retained_hot_bytes;
        let equality_cache_unique_bytes = product(&[HAMMING_V2_FIELD_BYTES, e_in + e_out])?;
        let equality_fully_issued_bytes = product(&[
            tiles,
            HAMMING_V2_FIELD_BYTES,
            rows.checked_add(e_out)
                .ok_or(HammingWeightV2Error::ArithmeticOverflow)?,
        ])?;
        let partial_write_read_bytes = product(&[
            2,
            HAMMING_V2_FIELD_BYTES,
            HAMMING_V2_SELECTORS as u64,
            HAMMING_V2_BINS as u64,
            e_out,
        ])?;
        let output_write_read_bytes = product(&[
            2,
            HAMMING_V2_FIELD_BYTES,
            HAMMING_V2_SELECTORS as u64,
            HAMMING_V2_BINS as u64,
        ])?;
        let accepted_cache_optimistic_bytes = sum(&[
            accepted_row_scan_bytes,
            equality_cache_unique_bytes,
            partial_write_read_bytes,
            output_write_read_bytes,
        ])?;
        let retained_cache_optimistic_bytes = sum(&[
            retained_consumer_read_bytes,
            equality_cache_unique_bytes,
            partial_write_read_bytes,
            output_write_read_bytes,
        ])?;
        let accepted_fully_issued_bytes = sum(&[
            accepted_row_scan_bytes,
            equality_fully_issued_bytes,
            partial_write_read_bytes,
            output_write_read_bytes,
        ])?;
        let retained_fully_issued_bytes = sum(&[
            retained_consumer_read_bytes,
            equality_fully_issued_bytes,
            partial_write_read_bytes,
            output_write_read_bytes,
        ])?;
        let fused_producer_plus_consumer_bytes = projection_write_bytes
            .checked_add(retained_cache_optimistic_bytes)
            .ok_or(HammingWeightV2Error::ArithmeticOverflow)?;
        let consumer_owned_bytes = geometry.buffer_lengths()?.consumer_owned_bytes()?;

        Ok(Self {
            rows,
            accepted_row_scan_bytes,
            projection_write_bytes,
            retained_hot_bytes,
            validity_bytes,
            retained_consumer_read_bytes,
            equality_cache_unique_bytes,
            equality_fully_issued_bytes,
            partial_write_read_bytes,
            output_write_read_bytes,
            accepted_cache_optimistic_bytes,
            retained_cache_optimistic_bytes,
            accepted_fully_issued_bytes,
            retained_fully_issued_bytes,
            fused_producer_plus_consumer_bytes,
            consumer_owned_bytes,
            retained_copy_floor_ns: rate_floor_ns(
                retained_cache_optimistic_bytes,
                M4_MAX_COPY_BYTES_PER_SECOND,
            ),
            retained_eighty_percent_copy_cap_ns: rate_floor_ns(
                retained_cache_optimistic_bytes,
                M4_MAX_COPY_BYTES_PER_SECOND * 4 / 5,
            ),
        })
    }

    pub fn cache_optimistic_reduction(self) -> f64 {
        self.accepted_cache_optimistic_bytes as f64 / self.retained_cache_optimistic_bytes as f64
    }

    pub fn producer_charged_reduction(self) -> f64 {
        self.accepted_cache_optimistic_bytes as f64 / self.fused_producer_plus_consumer_bytes as f64
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingWeightLatencyControl {
    pub cpu_member_ns: u64,
    pub metal_member_ns: u64,
    pub gpu_active_ns: u64,
    pub non_gpu_ns: u64,
    pub five_x_member_cap_ns: u64,
    pub robust_member_cap_ns: u64,
    pub five_x_active_cap_ns: u64,
    pub robust_active_cap_ns: u64,
}

impl HammingWeightLatencyControl {
    pub fn current_accepted() -> Self {
        let five_x_member_cap_ns = member_cap_ns(
            CURRENT_ACCEPTED_CPU_MEMBER_NS,
            HARD_FLOOR_NUMERATOR,
            HARD_FLOOR_DENOMINATOR,
        );
        let robust_member_cap_ns = member_cap_ns(
            CURRENT_ACCEPTED_CPU_MEMBER_NS,
            ROBUST_BAR_NUMERATOR,
            ROBUST_BAR_DENOMINATOR,
        );
        Self {
            cpu_member_ns: CURRENT_ACCEPTED_CPU_MEMBER_NS,
            metal_member_ns: CURRENT_ACCEPTED_METAL_MEMBER_NS,
            gpu_active_ns: CURRENT_ACCEPTED_GPU_ACTIVE_NS,
            non_gpu_ns: CURRENT_ACCEPTED_NON_GPU_NS,
            five_x_member_cap_ns,
            robust_member_cap_ns,
            five_x_active_cap_ns: five_x_member_cap_ns - CURRENT_ACCEPTED_NON_GPU_NS,
            robust_active_cap_ns: robust_member_cap_ns - CURRENT_ACCEPTED_NON_GPU_NS,
        }
    }

    pub const fn observed_speedup(self) -> RationalSpeedup {
        RationalSpeedup::new(self.cpu_member_ns, self.metal_member_ns)
    }

    pub fn robust_active_reduction_fraction(self) -> f64 {
        1.0 - self.robust_active_cap_ns as f64 / self.gpu_active_ns as f64
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingWeightWorkRoof {
    pub selector_row_opportunities: u64,
    pub retained_nonzero_adds: u64,
    pub robust_atomic_service_control_ns: u64,
    pub control_adds_per_second: u64,
    pub required_adds_per_second_for_robust_bar: u64,
    pub service_control_plus_current_remainder_ns: u64,
}

impl HammingWeightWorkRoof {
    pub fn frozen(control: HammingWeightLatencyControl) -> Self {
        Self {
            selector_row_opportunities: FROZEN_SELECTOR_ROW_OPPORTUNITIES,
            retained_nonzero_adds: FROZEN_RETAINED_NONZERO_ADDS,
            robust_atomic_service_control_ns: ROBUST_ATOMIC_SERVICE_CONTROL_NS,
            control_adds_per_second: work_rate_floor(
                FROZEN_RETAINED_NONZERO_ADDS,
                ROBUST_ATOMIC_SERVICE_CONTROL_NS,
            ),
            required_adds_per_second_for_robust_bar: work_rate_ceil(
                FROZEN_RETAINED_NONZERO_ADDS,
                control.robust_active_cap_ns,
            ),
            service_control_plus_current_remainder_ns: ROBUST_ATOMIC_SERVICE_CONTROL_NS
                + control.non_gpu_ns,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RationalSpeedup {
    cpu_ns: u64,
    metal_ns: u64,
}

impl RationalSpeedup {
    pub const fn new(cpu_ns: u64, metal_ns: u64) -> Self {
        Self { cpu_ns, metal_ns }
    }

    pub const fn cpu_ns(self) -> u64 {
        self.cpu_ns
    }

    pub const fn metal_ns(self) -> u64 {
        self.metal_ns
    }

    pub fn clears(self, numerator: u64, denominator: u64) -> bool {
        self.metal_ns != 0
            && u128::from(self.cpu_ns) * u128::from(denominator)
                >= u128::from(self.metal_ns) * u128::from(numerator)
    }

    pub fn as_f64(self) -> f64 {
        self.cpu_ns as f64 / self.metal_ns as f64
    }
}

impl Ord for RationalSpeedup {
    fn cmp(&self, other: &Self) -> Ordering {
        (u128::from(self.cpu_ns) * u128::from(other.metal_ns))
            .cmp(&(u128::from(other.cpu_ns) * u128::from(self.metal_ns)))
    }
}

impl PartialOrd for RationalSpeedup {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CampaignOrder {
    OptimizedFirst,
    RetainedFirst,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingWeightPairSample {
    pub order: CampaignOrder,
    pub cpu_member_ns: u64,
    pub retained_member_ns: u64,
    pub host_fiat_shamir_rounds: u32,
    pub proof_verified: bool,
    pub transcript_exact: bool,
    pub receipt_exact: bool,
    pub complete_member_accounting: bool,
}

impl HammingWeightPairSample {
    pub const fn speedup(self) -> RationalSpeedup {
        RationalSpeedup::new(self.cpu_member_ns, self.retained_member_ns)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingWeightCampaignSummary {
    pub minimum: RationalSpeedup,
    pub median: RationalSpeedup,
    pub optimized_first_median: RationalSpeedup,
    pub retained_first_median: RationalSpeedup,
}

pub fn evaluate_campaign(
    pairs: &[HammingWeightPairSample],
) -> Result<HammingWeightCampaignSummary, HammingWeightV2Error> {
    if pairs.len() != 5 {
        return Err(HammingWeightV2Error::CampaignLength(pairs.len()));
    }
    for (index, pair) in pairs.iter().copied().enumerate() {
        let expected_order = if index.is_multiple_of(2) {
            CampaignOrder::OptimizedFirst
        } else {
            CampaignOrder::RetainedFirst
        };
        if pair.order != expected_order {
            return Err(HammingWeightV2Error::CampaignOrder { index });
        }
        for (guard, exact) in [
            ("positive CPU member", pair.cpu_member_ns != 0),
            ("positive retained member", pair.retained_member_ns != 0),
            ("host Fiat-Shamir rounds", pair.host_fiat_shamir_rounds == 8),
            ("proof verification", pair.proof_verified),
            ("transcript parity", pair.transcript_exact),
            ("resource receipt", pair.receipt_exact),
            (
                "complete-member accounting",
                pair.complete_member_accounting,
            ),
        ] {
            if !exact {
                return Err(HammingWeightV2Error::CampaignGuard { index, guard });
            }
        }
        if !pair
            .speedup()
            .clears(HARD_FLOOR_NUMERATOR, HARD_FLOOR_DENOMINATOR)
        {
            return Err(HammingWeightV2Error::PairBelowFloor { index });
        }
    }

    let all = pairs
        .iter()
        .copied()
        .map(HammingWeightPairSample::speedup)
        .collect::<Vec<_>>();
    let optimized_first = pairs
        .iter()
        .step_by(2)
        .copied()
        .map(HammingWeightPairSample::speedup)
        .collect::<Vec<_>>();
    let retained_first = pairs
        .iter()
        .skip(1)
        .step_by(2)
        .copied()
        .map(HammingWeightPairSample::speedup)
        .collect::<Vec<_>>();
    let minimum = *all
        .iter()
        .min()
        .ok_or(HammingWeightV2Error::CampaignLength(0))?;
    let paired_median = median(all);
    let optimized_first_median = median(optimized_first);
    let retained_first_median = median(retained_first);
    for (name, speedup) in [
        ("paired median", paired_median),
        ("optimized-first stratum", optimized_first_median),
        ("retained-first stratum", retained_first_median),
    ] {
        if !speedup.clears(ROBUST_BAR_NUMERATOR, ROBUST_BAR_DENOMINATOR) {
            return Err(HammingWeightV2Error::CampaignBelowRobustBar(name));
        }
    }
    Ok(HammingWeightCampaignSummary {
        minimum,
        median: paired_median,
        optimized_first_median,
        retained_first_median,
    })
}

fn median(mut values: Vec<RationalSpeedup>) -> RationalSpeedup {
    values.sort_unstable();
    values[(values.len() - 1) / 2]
}

fn member_cap_ns(cpu_ns: u64, numerator: u64, denominator: u64) -> u64 {
    ((u128::from(cpu_ns) * u128::from(denominator)) / u128::from(numerator)) as u64
}

fn product(values: &[u64]) -> Result<u64, HammingWeightV2Error> {
    values.iter().try_fold(1u64, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(HammingWeightV2Error::ArithmeticOverflow)
    })
}

fn sum(values: &[u64]) -> Result<u64, HammingWeightV2Error> {
    values.iter().try_fold(0u64, |sum, value| {
        sum.checked_add(*value)
            .ok_or(HammingWeightV2Error::ArithmeticOverflow)
    })
}

fn rate_floor_ns(bytes: u64, bytes_per_second: u64) -> u64 {
    (u128::from(bytes) * 1_000_000_000).div_ceil(u128::from(bytes_per_second)) as u64
}

fn work_rate_floor(work: u64, duration_ns: u64) -> u64 {
    (u128::from(work) * 1_000_000_000 / u128::from(duration_ns)) as u64
}

fn work_rate_ceil(work: u64, duration_ns: u64) -> u64 {
    (u128::from(work) * 1_000_000_000).div_ceil(u128::from(duration_ns)) as u64
}
