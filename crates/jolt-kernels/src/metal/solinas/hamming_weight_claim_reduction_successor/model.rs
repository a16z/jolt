//! Pure analytical model for the retained all-hot projection schedule.

pub const LOG_T: usize = 26;
pub const ROWS: u64 = 1 << LOG_T;
pub const SELECTORS: u64 = 29;
pub const BINS: u64 = 256;
pub const FIELD_BYTES: u64 = 16;
pub const RESIDENT_ROW_BYTES: u64 = 40;
pub const PROJECTION_HOT_PLANES: u64 = 29;
pub const PROJECTION_FLAG_PLANES: u64 = 1;
pub const PROJECTION_PLANES: u64 = PROJECTION_HOT_PLANES + PROJECTION_FLAG_PLANES;
pub const INNER_LOG2: usize = 15;
pub const E_IN_ELEMENTS: u64 = 1 << INNER_LOG2;
pub const E_OUT_ELEMENTS: u64 = ROWS / E_IN_ELEMENTS;
pub const TILE_SELECTORS: [u64; 5] = [6, 6, 6, 6, 5];
pub const DEVICE_MAX_BUFFER_BYTES: u64 = 86_586_540_032;

pub const COPY_BYTES_PER_SECOND: u64 = 451_701_710_520;
pub const FROZEN_RETAINED_ADDS: u64 = 1_588_505_707;
pub const FAST_STANDALONE_RETAINED_ADDS_PER_SECOND: u64 = 47_405_000_000;
pub const ROBUST_STANDALONE_GPU_ACTIVE_NS: u64 = 35_451_625;

pub const CPU_EQUAL_INPUT_SAMPLES_NS: [u64; 5] = [
    545_613_583,
    554_614_169,
    525_892_210,
    548_702_500,
    555_909_956,
];
pub const METAL_MEMBER_SAMPLES_NS: [u64; 5] = [
    112_953_333,
    113_150_665,
    111_646_165,
    110_735_835,
    110_867_875,
];
pub const METAL_GPU_ACTIVE_SAMPLES_NS: [u64; 5] =
    [84_675_875, 85_104_750, 84_896_875, 84_738_625, 84_998_875];

pub const CPU_EQUAL_INPUT_MEDIAN_NS: u64 = 548_702_500;
pub const METAL_MEMBER_MEDIAN_NS: u64 = 111_646_165;
pub const METAL_GPU_ACTIVE_MEDIAN_NS: u64 = 84_896_875;
pub const METAL_NON_GPU_MEDIAN_NS: u64 = 26_749_290;
pub const FIVE_X_MEMBER_CAP_NS: u64 = CPU_EQUAL_INPUT_MEDIAN_NS / 5;
pub const EIGHT_X_MEMBER_CAP_NS: u64 = CPU_EQUAL_INPUT_MEDIAN_NS / 8;
pub const FIVE_X_ACTIVE_CAP_NS: u64 = FIVE_X_MEMBER_CAP_NS - METAL_NON_GPU_MEDIAN_NS;
pub const EIGHT_X_ACTIVE_CAP_NS: u64 = EIGHT_X_MEMBER_CAP_NS - METAL_NON_GPU_MEDIAN_NS;
pub const FIRST_PROBE_ACTIVE_TARGET_NS: u64 = 40_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TrafficModel {
    pub accepted_row_read_bytes: u64,
    pub projection_write_bytes: u64,
    pub projection_retained_hot_bytes: u64,
    pub projection_flag_bytes: u64,
    pub projection_increment_over_tail_only_bytes: u64,
    pub successor_projection_read_bytes: u64,
    pub equality_cache_unique_bytes: u64,
    pub equality_fully_issued_bytes: u64,
    pub partial_write_read_bytes: u64,
    pub output_write_read_bytes: u64,
    pub accepted_cache_optimistic_bytes: u64,
    pub successor_cache_optimistic_bytes: u64,
    pub accepted_fully_issued_bytes: u64,
    pub successor_fully_issued_bytes: u64,
    pub fused_producer_plus_consumer_bytes: u64,
    pub successor_copy_floor_ns: u64,
    pub successor_eighty_percent_copy_cap_ns: u64,
}

impl TrafficModel {
    pub fn log_26() -> Self {
        let accepted_row_read_bytes = 5 * RESIDENT_ROW_BYTES * ROWS;
        let projection_write_bytes = PROJECTION_PLANES * ROWS;
        let projection_retained_hot_bytes = PROJECTION_HOT_PLANES * ROWS;
        let projection_flag_bytes = PROJECTION_FLAG_PLANES * ROWS;
        let projection_increment_over_tail_only_bytes = 6 * ROWS;
        let successor_projection_read_bytes = PROJECTION_HOT_PLANES * ROWS;
        let equality_cache_unique_bytes = FIELD_BYTES * (E_IN_ELEMENTS + E_OUT_ELEMENTS);
        let equality_fully_issued_bytes = 5 * FIELD_BYTES * ROWS + 5 * FIELD_BYTES * E_OUT_ELEMENTS;
        let partial_write_read_bytes = 2 * FIELD_BYTES * SELECTORS * BINS * E_OUT_ELEMENTS;
        let output_write_read_bytes = 2 * FIELD_BYTES * SELECTORS * BINS;
        let accepted_cache_optimistic_bytes = accepted_row_read_bytes
            + equality_cache_unique_bytes
            + partial_write_read_bytes
            + output_write_read_bytes;
        let successor_cache_optimistic_bytes = successor_projection_read_bytes
            + equality_cache_unique_bytes
            + partial_write_read_bytes
            + output_write_read_bytes;
        let accepted_fully_issued_bytes = accepted_row_read_bytes
            + equality_fully_issued_bytes
            + partial_write_read_bytes
            + output_write_read_bytes;
        let successor_fully_issued_bytes = successor_projection_read_bytes
            + equality_fully_issued_bytes
            + partial_write_read_bytes
            + output_write_read_bytes;

        Self {
            accepted_row_read_bytes,
            projection_write_bytes,
            projection_retained_hot_bytes,
            projection_flag_bytes,
            projection_increment_over_tail_only_bytes,
            successor_projection_read_bytes,
            equality_cache_unique_bytes,
            equality_fully_issued_bytes,
            partial_write_read_bytes,
            output_write_read_bytes,
            accepted_cache_optimistic_bytes,
            successor_cache_optimistic_bytes,
            accepted_fully_issued_bytes,
            successor_fully_issued_bytes,
            fused_producer_plus_consumer_bytes: projection_write_bytes
                + successor_cache_optimistic_bytes,
            successor_copy_floor_ns: rate_floor_ns(
                successor_cache_optimistic_bytes,
                COPY_BYTES_PER_SECOND,
            ),
            successor_eighty_percent_copy_cap_ns: rate_floor_ns(
                successor_cache_optimistic_bytes,
                COPY_BYTES_PER_SECOND * 4 / 5,
            ),
        }
    }

    pub fn cache_optimistic_reduction(self) -> f64 {
        self.accepted_cache_optimistic_bytes as f64 / self.successor_cache_optimistic_bytes as f64
    }

    pub fn fused_producer_reduction(self) -> f64 {
        self.accepted_cache_optimistic_bytes as f64 / self.fused_producer_plus_consumer_bytes as f64
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoofModel {
    pub copy_floor_ns: u64,
    pub fast_standalone_service_ns: u64,
    pub robust_standalone_service_ns: u64,
    pub directional_screen_ns: u64,
    pub eighty_percent_fast_service_ns: u64,
}

impl RoofModel {
    pub fn frozen_screen(traffic: TrafficModel) -> Self {
        let fast_standalone_service_ns = rate_floor_ns(
            FROZEN_RETAINED_ADDS,
            FAST_STANDALONE_RETAINED_ADDS_PER_SECOND,
        );
        Self {
            copy_floor_ns: traffic.successor_copy_floor_ns,
            fast_standalone_service_ns,
            robust_standalone_service_ns: ROBUST_STANDALONE_GPU_ACTIVE_NS,
            directional_screen_ns: traffic
                .successor_copy_floor_ns
                .max(fast_standalone_service_ns),
            eighty_percent_fast_service_ns: rate_floor_ns(
                FROZEN_RETAINED_ADDS,
                FAST_STANDALONE_RETAINED_ADDS_PER_SECOND * 4 / 5,
            ),
        }
    }

    pub fn required_retained_add_fraction_for(active_ns: u64) -> f64 {
        let required = FROZEN_RETAINED_ADDS as f64 * 1e9 / active_ns as f64;
        required / FAST_STANDALONE_RETAINED_ADDS_PER_SECOND as f64
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FirstProbeDecision {
    PursueEightX,
    OneCounterDirectedIteration,
    KillTopology,
}

pub const fn first_probe_decision(gpu_active_ns: u64) -> FirstProbeDecision {
    if gpu_active_ns <= EIGHT_X_ACTIVE_CAP_NS {
        FirstProbeDecision::PursueEightX
    } else if gpu_active_ns < METAL_GPU_ACTIVE_MEDIAN_NS {
        FirstProbeDecision::OneCounterDirectedIteration
    } else {
        FirstProbeDecision::KillTopology
    }
}

fn rate_floor_ns(work: u64, rate_per_second: u64) -> u64 {
    let numerator = u128::from(work) * 1_000_000_000;
    numerator.div_ceil(u128::from(rate_per_second)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_26_traffic_ledger_is_exact() {
        let traffic = TrafficModel::log_26();

        assert_eq!(TILE_SELECTORS.iter().sum::<u64>(), SELECTORS);
        assert_eq!(E_OUT_ELEMENTS, 2_048);
        assert_eq!(traffic.accepted_row_read_bytes, 13_421_772_800);
        assert_eq!(traffic.projection_write_bytes, 2_013_265_920);
        assert_eq!(traffic.projection_retained_hot_bytes, 1_946_157_056);
        assert_eq!(traffic.projection_flag_bytes, 67_108_864);
        assert_eq!(
            traffic.projection_increment_over_tail_only_bytes,
            402_653_184
        );
        assert_eq!(traffic.successor_projection_read_bytes, 1_946_157_056);
        assert_eq!(traffic.equality_cache_unique_bytes, 557_056);
        assert_eq!(traffic.partial_write_read_bytes, 486_539_264);
        assert_eq!(traffic.output_write_read_bytes, 237_568);
        assert_eq!(traffic.accepted_cache_optimistic_bytes, 13_909_106_688);
        assert_eq!(traffic.successor_cache_optimistic_bytes, 2_433_490_944);
        assert_eq!(traffic.accepted_fully_issued_bytes, 19_277_422_592);
        assert_eq!(traffic.successor_fully_issued_bytes, 7_801_806_848);
        assert_eq!(traffic.fused_producer_plus_consumer_bytes, 4_446_756_864);
        assert_eq!(traffic.successor_copy_floor_ns, 5_387_385);
        assert_eq!(traffic.successor_eighty_percent_copy_cap_ns, 6_734_232);
        assert!(traffic.cache_optimistic_reduction() > 5.71);
        assert!(traffic.fused_producer_reduction() > 3.12);
        const {
            assert!(PROJECTION_HOT_PLANES * (1u64 << 28) < DEVICE_MAX_BUFFER_BYTES);
        }
    }

    #[test]
    fn frozen_latency_budget_exposes_the_eight_x_bar() {
        assert_eq!(FIVE_X_MEMBER_CAP_NS, 109_740_500);
        assert_eq!(EIGHT_X_MEMBER_CAP_NS, 68_587_812);
        assert_eq!(FIVE_X_ACTIVE_CAP_NS, 82_991_210);
        assert_eq!(EIGHT_X_ACTIVE_CAP_NS, 41_838_522);
        assert_eq!(METAL_MEMBER_MEDIAN_NS - FIVE_X_MEMBER_CAP_NS, 1_905_665);
        const {
            assert!(METAL_NON_GPU_MEDIAN_NS + FIRST_PROBE_ACTIVE_TARGET_NS < EIGHT_X_MEMBER_CAP_NS);
        }
    }

    #[test]
    fn retained_add_control_keeps_eight_x_credible_but_not_free() {
        let roof = RoofModel::frozen_screen(TrafficModel::log_26());

        assert_eq!(roof.copy_floor_ns, 5_387_385);
        assert_eq!(roof.fast_standalone_service_ns, 33_509_244);
        assert_eq!(roof.robust_standalone_service_ns, 35_451_625);
        assert_eq!(roof.directional_screen_ns, 33_509_244);
        assert_eq!(roof.eighty_percent_fast_service_ns, 41_886_555);
        let required = RoofModel::required_retained_add_fraction_for(EIGHT_X_ACTIVE_CAP_NS);
        assert!((required - 0.800_918).abs() < 0.000_01);
    }

    #[test]
    fn first_probe_has_one_bounded_iteration_lane() {
        assert_eq!(
            first_probe_decision(FIRST_PROBE_ACTIVE_TARGET_NS),
            FirstProbeDecision::PursueEightX
        );
        assert_eq!(
            first_probe_decision(60_000_000),
            FirstProbeDecision::OneCounterDirectedIteration
        );
        assert_eq!(
            first_probe_decision(METAL_GPU_ACTIVE_MEDIAN_NS),
            FirstProbeDecision::KillTopology
        );
    }
}
