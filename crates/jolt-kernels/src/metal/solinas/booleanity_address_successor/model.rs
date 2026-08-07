//! Analytical traffic, work, topology, and speedup gates.
//!
//! The traffic floor uses the retained M4 Max copy control. Cached equality
//! reads and threadgroup atomics need separate same-binary controls before the
//! 80%-of-roof screen can be evaluated; this model keeps those terms explicit.

use super::{
    BooleanityAddressSuccessorError, BooleanityAddressSuccessorGeometry,
    BOOLEANITY_ADDRESS_SUCCESSOR_ACCUMULATOR_THREADS, BOOLEANITY_ADDRESS_SUCCESSOR_BINS,
    BOOLEANITY_ADDRESS_SUCCESSOR_FIRST_TILE_SELECTORS, BOOLEANITY_ADDRESS_SUCCESSOR_PACKED_PLANES,
    BOOLEANITY_ADDRESS_SUCCESSOR_REMAINING_TILES, BOOLEANITY_ADDRESS_SUCCESSOR_SELECTORS,
    BOOLEANITY_ADDRESS_SUCCESSOR_SIMD_WIDTH,
};

pub const LOG_T_26: usize = 26;
pub const ROWS_AT_LOG_T_26: u64 = 1 << LOG_T_26;
pub const COPY_GIB_PER_SECOND: f64 = 420.68;
pub const FIELD_PRODUCTS_PER_SECOND: f64 = 16.42e9;

pub const CPU_EQUAL_INPUT_SAMPLES_NS: [u64; 5] = [
    972_037_919,
    929_139_914,
    899_211_126,
    907_191_128,
    948_932_957,
];
pub const CPU_EQUAL_INPUT_MEDIAN_NS: u64 = 929_139_914;
pub const ACCEPTED_METAL_SAMPLES_NS: [u64; 5] = [
    111_635_498,
    121_546_876,
    106_372_335,
    128_948_627,
    110_591_793,
];
pub const ACCEPTED_METAL_MEDIAN_NS: u64 = 111_635_498;

const FIELD_BYTES: u64 = 16;
const ROW_BYTES: u64 = 40;
const PACKED_BYTES_PER_ROW: u64 = BOOLEANITY_ADDRESS_SUCCESSOR_PACKED_PLANES as u64;
const PACKED_READ_BYTES_PER_ROW: u64 = 25;
const E_IN_PASSES: u64 = 1 + BOOLEANITY_ADDRESS_SUCCESSOR_REMAINING_TILES as u64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpeedupGate {
    pub speedup: u64,
    pub complete_member_cap_ns: u64,
}

pub const FIVE_X_GATE: SpeedupGate = SpeedupGate {
    speedup: 5,
    complete_member_cap_ns: CPU_EQUAL_INPUT_MEDIAN_NS / 5,
};

pub const EIGHT_X_GATE: SpeedupGate = SpeedupGate {
    speedup: 8,
    complete_member_cap_ns: CPU_EQUAL_INPUT_MEDIAN_NS / 8,
};

pub const TEN_X_STRETCH_GATE: SpeedupGate = SpeedupGate {
    speedup: 10,
    complete_member_cap_ns: CPU_EQUAL_INPUT_MEDIAN_NS / 10,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LaneTopology {
    pub simd_width: u64,
    pub row_lanes_per_simd: u64,
    pub first_tile_selectors_per_row_lane: u64,
    pub remaining_tile_groups_per_outer_block: u64,
    pub bucket_owner_lanes: u64,
}

pub const LANE_TOPOLOGY: LaneTopology = LaneTopology {
    simd_width: BOOLEANITY_ADDRESS_SUCCESSOR_SIMD_WIDTH as u64,
    row_lanes_per_simd: BOOLEANITY_ADDRESS_SUCCESSOR_SIMD_WIDTH as u64,
    first_tile_selectors_per_row_lane: BOOLEANITY_ADDRESS_SUCCESSOR_FIRST_TILE_SELECTORS as u64,
    remaining_tile_groups_per_outer_block: BOOLEANITY_ADDRESS_SUCCESSOR_REMAINING_TILES as u64,
    bucket_owner_lanes: 0,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TrafficModel {
    pub resident_row_read_bytes: u64,
    pub packed_write_bytes: u64,
    pub packed_read_bytes: u64,
    pub e_in_cache_unique_bytes: u64,
    pub e_in_issued_bytes: u64,
    pub e_out_cache_unique_bytes: u64,
    pub e_out_issued_bytes: u64,
    pub partial_write_read_bytes: u64,
    pub output_write_read_bytes: u64,
    pub pack_and_first_cache_optimistic_bytes: u64,
    pub packed_tiles_cache_optimistic_bytes: u64,
    pub finalize_cache_optimistic_bytes: u64,
    pub pack_and_first_e_in_issued_bytes: u64,
    pub packed_tiles_e_in_issued_bytes: u64,
    pub compulsory_unique_bytes: u64,
    pub cache_optimistic_bytes: u64,
    pub fully_issued_bytes: u64,
    pub cache_optimistic_copy_floor_ns: u64,
    pub cache_optimistic_eighty_percent_cap_ns: u64,
    pub accepted_cache_optimistic_bytes: u64,
}

impl TrafficModel {
    pub fn new(
        geometry: BooleanityAddressSuccessorGeometry,
    ) -> Result<Self, BooleanityAddressSuccessorError> {
        let rows = geometry.rows() as u64;
        let e_in = geometry.e_in_length() as u64;
        let e_out = geometry.e_out_length() as u64;
        let fields = checked_mul(
            BOOLEANITY_ADDRESS_SUCCESSOR_SELECTORS as u64,
            BOOLEANITY_ADDRESS_SUCCESSOR_BINS as u64,
        )?;

        let resident_row_read_bytes = checked_mul(ROW_BYTES, rows)?;
        let packed_write_bytes = checked_mul(PACKED_BYTES_PER_ROW, rows)?;
        let packed_read_bytes = checked_mul(PACKED_READ_BYTES_PER_ROW, rows)?;
        let e_in_cache_unique_bytes = checked_mul(FIELD_BYTES, e_in)?;
        let e_in_issued_bytes = checked_mul(checked_mul(FIELD_BYTES, rows)?, E_IN_PASSES)?;
        let e_out_cache_unique_bytes = checked_mul(FIELD_BYTES, e_out)?;
        let e_out_issued_bytes = checked_mul(
            checked_mul(FIELD_BYTES, e_out)?,
            1 + BOOLEANITY_ADDRESS_SUCCESSOR_REMAINING_TILES as u64,
        )?;
        let partial_bytes = checked_mul(checked_mul(FIELD_BYTES, fields)?, e_out)?;
        let first_partial_bytes = checked_mul(
            checked_mul(
                FIELD_BYTES,
                BOOLEANITY_ADDRESS_SUCCESSOR_FIRST_TILE_SELECTORS as u64
                    * BOOLEANITY_ADDRESS_SUCCESSOR_BINS as u64,
            )?,
            e_out,
        )?;
        let remaining_partial_bytes = partial_bytes
            .checked_sub(first_partial_bytes)
            .ok_or(BooleanityAddressSuccessorError::ArithmeticOverflow)?;
        let partial_write_read_bytes = checked_mul(2, partial_bytes)?;
        let output_write_read_bytes = checked_mul(2, checked_mul(FIELD_BYTES, fields)?)?;
        let pack_and_first_e_in_issued_bytes = checked_mul(FIELD_BYTES, rows)?;
        let packed_tiles_e_in_issued_bytes = checked_mul(
            pack_and_first_e_in_issued_bytes,
            BOOLEANITY_ADDRESS_SUCCESSOR_REMAINING_TILES as u64,
        )?;
        let pack_and_first_cache_optimistic_bytes = checked_sum(&[
            resident_row_read_bytes,
            packed_write_bytes,
            first_partial_bytes,
            e_in_cache_unique_bytes,
            e_out_cache_unique_bytes,
        ])?;
        let packed_tiles_cache_optimistic_bytes =
            checked_sum(&[packed_read_bytes, remaining_partial_bytes])?;
        let finalize_cache_optimistic_bytes =
            checked_sum(&[partial_bytes, output_write_read_bytes])?;

        let compulsory_unique_bytes = checked_sum(&[
            resident_row_read_bytes,
            packed_write_bytes,
            partial_bytes,
            output_write_read_bytes / 2,
            e_in_cache_unique_bytes,
            e_out_cache_unique_bytes,
        ])?;
        let cache_optimistic_bytes = checked_sum(&[
            resident_row_read_bytes,
            packed_write_bytes,
            packed_read_bytes,
            e_in_cache_unique_bytes,
            e_out_cache_unique_bytes,
            partial_write_read_bytes,
            output_write_read_bytes,
        ])?;
        let fully_issued_bytes = checked_sum(&[
            resident_row_read_bytes,
            packed_write_bytes,
            packed_read_bytes,
            e_in_issued_bytes,
            e_out_issued_bytes,
            partial_write_read_bytes,
            output_write_read_bytes,
        ])?;
        let accepted_cache_optimistic_bytes = checked_sum(&[
            checked_mul(5 * ROW_BYTES, rows)?,
            e_in_cache_unique_bytes,
            e_out_cache_unique_bytes,
            partial_write_read_bytes,
            output_write_read_bytes,
        ])?;
        let cache_optimistic_copy_floor_ns =
            gib_floor_ns(cache_optimistic_bytes, COPY_GIB_PER_SECOND);

        Ok(Self {
            resident_row_read_bytes,
            packed_write_bytes,
            packed_read_bytes,
            e_in_cache_unique_bytes,
            e_in_issued_bytes,
            e_out_cache_unique_bytes,
            e_out_issued_bytes,
            partial_write_read_bytes,
            output_write_read_bytes,
            pack_and_first_cache_optimistic_bytes,
            packed_tiles_cache_optimistic_bytes,
            finalize_cache_optimistic_bytes,
            pack_and_first_e_in_issued_bytes,
            packed_tiles_e_in_issued_bytes,
            compulsory_unique_bytes,
            cache_optimistic_bytes,
            fully_issued_bytes,
            cache_optimistic_copy_floor_ns,
            cache_optimistic_eighty_percent_cap_ns: cache_optimistic_copy_floor_ns
                .saturating_mul(5)
                / 4,
            accepted_cache_optimistic_bytes,
        })
    }

    pub fn large_state_reduction_ratio(self) -> f64 {
        self.accepted_cache_optimistic_bytes as f64 / self.cache_optimistic_bytes as f64
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkloadCensus {
    pub rows: u64,
    pub bytecode_present_rows: u64,
    pub ram_present_rows: u64,
    /// Rows whose recentered selectors 24, 25, and 26 all target bucket zero.
    pub common_high_increment_rows: u64,
}

impl WorkloadCensus {
    pub fn validate(self) -> Result<Self, BooleanityAddressSuccessorError> {
        if self.rows == 0 {
            return Err(BooleanityAddressSuccessorError::InvalidCensus {
                name: "rows",
                rows: self.rows,
                got: self.rows,
            });
        }
        for (name, got) in [
            ("bytecode present", self.bytecode_present_rows),
            ("RAM present", self.ram_present_rows),
            ("common high increment", self.common_high_increment_rows),
        ] {
            if got > self.rows {
                return Err(BooleanityAddressSuccessorError::InvalidCensus {
                    name,
                    rows: self.rows,
                    got,
                });
            }
        }
        Ok(self)
    }

    pub const fn dense(rows: u64) -> Self {
        Self {
            rows,
            bytecode_present_rows: rows,
            ram_present_rows: rows,
            common_high_increment_rows: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkModel {
    pub selector_row_opportunities: u64,
    pub present_field_contributions: u64,
    pub local_field_additions: u64,
    pub atomic_field_additions: u64,
    pub four_limb_atomic_word_adds: u64,
    pub overflow_atomic_word_adds_upper_bound: u64,
    pub bucket_products: u64,
    pub bucket_product_floor_ns: u64,
    pub pack_and_first_atomic_field_additions: u64,
    pub packed_tiles_atomic_field_additions: u64,
    pub pack_and_first_atomic_word_adds: u64,
    pub packed_tiles_atomic_word_adds: u64,
    pub pack_and_first_bucket_products: u64,
    pub packed_tiles_bucket_products: u64,
}

impl WorkModel {
    pub fn new(
        census: WorkloadCensus,
        geometry: BooleanityAddressSuccessorGeometry,
    ) -> Result<Self, BooleanityAddressSuccessorError> {
        let census = census.validate()?;
        if census.rows != geometry.rows() as u64 {
            return Err(BooleanityAddressSuccessorError::InvalidCensus {
                name: "geometry rows",
                rows: geometry.rows() as u64,
                got: census.rows,
            });
        }
        let selector_row_opportunities =
            checked_mul(BOOLEANITY_ADDRESS_SUCCESSOR_SELECTORS as u64, census.rows)?;
        // Sixteen lookup, eight increment-byte, and one carry columns are
        // always present. Bytecode and RAM each contribute two optional rows.
        let present_field_contributions = checked_sum(&[
            checked_mul(25, census.rows)?,
            checked_mul(2, census.bytecode_present_rows)?,
            checked_mul(2, census.ram_present_rows)?,
        ])?;
        let workers = checked_mul(
            BOOLEANITY_ADDRESS_SUCCESSOR_ACCUMULATOR_THREADS as u64,
            geometry.e_out_length() as u64,
        )?;
        let local_flushes = checked_mul(6, workers)?;
        let replaced_row_atomics = checked_sum(&[
            checked_mul(3, census.common_high_increment_rows)?,
            census.rows,
        ])?;
        let atomic_field_additions = present_field_contributions
            .checked_sub(replaced_row_atomics)
            .and_then(|value| value.checked_add(local_flushes))
            .ok_or(BooleanityAddressSuccessorError::ArithmeticOverflow)?;
        let bucket_products = checked_mul(
            checked_mul(
                BOOLEANITY_ADDRESS_SUCCESSOR_SELECTORS as u64,
                BOOLEANITY_ADDRESS_SUCCESSOR_BINS as u64,
            )?,
            geometry.e_out_length() as u64,
        )?;
        let pack_and_first_atomic_field_additions = checked_mul(
            BOOLEANITY_ADDRESS_SUCCESSOR_FIRST_TILE_SELECTORS as u64,
            census.rows,
        )?;
        let pack_and_first_atomic_word_adds =
            checked_mul(4, pack_and_first_atomic_field_additions)?;
        let four_limb_atomic_word_adds = checked_mul(4, atomic_field_additions)?;
        let packed_tiles_atomic_field_additions = atomic_field_additions
            .checked_sub(pack_and_first_atomic_field_additions)
            .ok_or(BooleanityAddressSuccessorError::ArithmeticOverflow)?;
        let packed_tiles_atomic_word_adds = four_limb_atomic_word_adds
            .checked_sub(pack_and_first_atomic_word_adds)
            .ok_or(BooleanityAddressSuccessorError::ArithmeticOverflow)?;
        let pack_and_first_bucket_products = checked_mul(
            checked_mul(
                BOOLEANITY_ADDRESS_SUCCESSOR_FIRST_TILE_SELECTORS as u64,
                BOOLEANITY_ADDRESS_SUCCESSOR_BINS as u64,
            )?,
            geometry.e_out_length() as u64,
        )?;
        let packed_tiles_bucket_products = bucket_products
            .checked_sub(pack_and_first_bucket_products)
            .ok_or(BooleanityAddressSuccessorError::ArithmeticOverflow)?;

        Ok(Self {
            selector_row_opportunities,
            present_field_contributions,
            local_field_additions: checked_sum(&[census.common_high_increment_rows, census.rows])?,
            atomic_field_additions,
            four_limb_atomic_word_adds,
            overflow_atomic_word_adds_upper_bound: atomic_field_additions,
            bucket_products,
            bucket_product_floor_ns: rate_floor_ns(bucket_products, FIELD_PRODUCTS_PER_SECOND),
            pack_and_first_atomic_field_additions,
            packed_tiles_atomic_field_additions,
            pack_and_first_atomic_word_adds,
            packed_tiles_atomic_word_adds,
            pack_and_first_bucket_products,
            packed_tiles_bucket_products,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeasuredControls {
    pub cached_read_gib_per_second: f64,
    /// Complete `solinas_deferred_atomic_add_5` calls per second, including
    /// the data-dependent fifth-word update.
    pub threadgroup_atomic_field_additions_per_second: f64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PhaseRoof {
    pub dram_floor_ns: u64,
    pub cached_weight_issue_floor_ns: u64,
    pub atomic_issue_floor_ns: u64,
    pub bucket_product_floor_ns: u64,
    pub lower_bound_ns: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CalibratedRoof {
    pub pack_and_first: PhaseRoof,
    pub packed_tiles: PhaseRoof,
    pub finalize: PhaseRoof,
    pub combined_lower_bound_ns: u64,
    pub eighty_percent_cap_ns: u64,
}

impl CalibratedRoof {
    pub fn new(traffic: TrafficModel, work: WorkModel, controls: MeasuredControls) -> Option<Self> {
        if !controls.cached_read_gib_per_second.is_finite()
            || controls.cached_read_gib_per_second <= 0.0
            || !controls
                .threadgroup_atomic_field_additions_per_second
                .is_finite()
            || controls.threadgroup_atomic_field_additions_per_second <= 0.0
        {
            return None;
        }
        let pack_and_first = phase_roof(
            traffic.pack_and_first_cache_optimistic_bytes,
            traffic.pack_and_first_e_in_issued_bytes,
            work.pack_and_first_atomic_field_additions,
            work.pack_and_first_bucket_products,
            controls,
        );
        let packed_tiles = phase_roof(
            traffic.packed_tiles_cache_optimistic_bytes,
            traffic.packed_tiles_e_in_issued_bytes,
            work.packed_tiles_atomic_field_additions,
            work.packed_tiles_bucket_products,
            controls,
        );
        let finalize = phase_roof(traffic.finalize_cache_optimistic_bytes, 0, 0, 0, controls);
        let combined_lower_bound_ns = pack_and_first
            .lower_bound_ns
            .saturating_add(packed_tiles.lower_bound_ns)
            .saturating_add(finalize.lower_bound_ns);
        Some(Self {
            pack_and_first,
            packed_tiles,
            finalize,
            combined_lower_bound_ns,
            eighty_percent_cap_ns: combined_lower_bound_ns.saturating_mul(5) / 4,
        })
    }
}

fn phase_roof(
    dram_bytes: u64,
    cached_weight_bytes: u64,
    atomic_field_additions: u64,
    bucket_products: u64,
    controls: MeasuredControls,
) -> PhaseRoof {
    let dram_floor_ns = gib_floor_ns(dram_bytes, COPY_GIB_PER_SECOND);
    let cached_weight_issue_floor_ns =
        gib_floor_ns(cached_weight_bytes, controls.cached_read_gib_per_second);
    let atomic_issue_floor_ns = rate_floor_ns(
        atomic_field_additions,
        controls.threadgroup_atomic_field_additions_per_second,
    );
    let bucket_product_floor_ns = rate_floor_ns(bucket_products, FIELD_PRODUCTS_PER_SECOND);
    let lower_bound_ns = [
        dram_floor_ns,
        cached_weight_issue_floor_ns,
        atomic_issue_floor_ns,
        bucket_product_floor_ns,
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    PhaseRoof {
        dram_floor_ns,
        cached_weight_issue_floor_ns,
        atomic_issue_floor_ns,
        bucket_product_floor_ns,
        lower_bound_ns,
    }
}

fn checked_mul(left: u64, right: u64) -> Result<u64, BooleanityAddressSuccessorError> {
    left.checked_mul(right)
        .ok_or(BooleanityAddressSuccessorError::ArithmeticOverflow)
}

fn checked_sum(values: &[u64]) -> Result<u64, BooleanityAddressSuccessorError> {
    values.iter().try_fold(0u64, |sum, value| {
        sum.checked_add(*value)
            .ok_or(BooleanityAddressSuccessorError::ArithmeticOverflow)
    })
}

fn gib_floor_ns(bytes: u64, gib_per_second: f64) -> u64 {
    ((bytes as f64 / (gib_per_second * (1u64 << 30) as f64)) * 1e9).ceil() as u64
}

fn rate_floor_ns(operations: u64, operations_per_second: f64) -> u64 {
    ((operations as f64 / operations_per_second) * 1e9).ceil() as u64
}
