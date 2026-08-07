//! Exact analytical model for the unregistered RAM-RA virtualization successor.

use core::fmt;

pub const TARGET_LOG_T: u32 = 26;
pub const TARGET_ROWS: u64 = 1 << TARGET_LOG_T;
pub const TARGET_ACCESSES: u64 = 22_000_000;
pub const ADDRESS_DOMAIN: u64 = 1 << 13;
pub const PREFIX_ROUNDS: usize = 5;
pub const TILE_WIDTH: u64 = 16;
pub const TARGET_TILES: u64 = TARGET_ROWS / TILE_WIDTH;
pub const SELECTED_LAST_METAL_MESSAGE: usize = 16;

pub const FROZEN_CPU_ARTIFACT: &str =
    "benchmark-runs/metal-piop-eval/20260806-133709-697013/result.json";
pub const FROZEN_CPU_REVISION: &str = "5f520c21e338632aa0bf5936ceb02be6c22fa40f";
pub const FROZEN_CPU_SAMPLES_NS: [u64; 5] = [
    278_459_584,
    332_764_663,
    270_177_247,
    274_665_791,
    270_797_830,
];
pub const FROZEN_CPU_MEDIAN_NS: u64 = 274_665_791;
pub const FIVE_X_CAP_NS: u64 = FROZEN_CPU_MEDIAN_NS / 5;
pub const EIGHT_X_CAP_NS: u64 = FROZEN_CPU_MEDIAN_NS / 8;
pub const CPU_CONTINUATION_AFTER_MESSAGE_16_SAMPLES_NS: [u64; 5] =
    [720_208, 1_004_040, 748_206, 566_458, 912_583];
pub const CPU_CONTINUATION_AFTER_MESSAGE_16_NS: u64 = 748_206;

pub const RETAINED_COPY_BYTES_PER_SECOND: u64 = 451_701_710_520;
pub const RETAINED_FIELD_PRODUCTS_PER_SECOND: u64 = 18_100_000_000;
pub const COMMAND_WALL_FLOOR_NS: u64 = 141_000;
pub const SEQUENCE_SETUP_FLOOR_NS: u64 = 141_000;
pub const ROOF_EFFICIENCY_PERMILLE: u64 = 800;
pub const COMPLETE_PURSUIT_BAR_NS: u64 = 25_000_000;

const FIELD_BYTES: u64 = 16;
const ADDRESS_BYTES: u64 = 2;
const MASK_BYTES: u64 = 2;
const OFFSET_BYTES: u64 = 4;
const MESSAGE_COLUMNS: u64 = 2;
const SIMD_WIDTH: u64 = 32;
const FACTORS: u64 = 2;
const BINS: u64 = 256;
const NANOS_PER_SECOND: u128 = 1_000_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefixCensus {
    /// `live_blocks[r]` is the number of live blocks after binding `r + 1` bits.
    pub live_blocks: [u64; PREFIX_ROUNDS],
    /// Number of occupied high-coordinate groups in message `r`.
    pub occupied_outer_groups: [u64; PREFIX_ROUNDS],
}

impl PrefixCensus {
    /// The placement-independent upper bound for a fixed access count.
    pub const fn target_worst_case() -> Self {
        Self {
            live_blocks: [22_000_000, 16_777_216, 8_388_608, 4_194_304, 2_097_152],
            occupied_outer_groups: [8_192; PREFIX_ROUNDS],
        }
    }

    pub fn validate(self, rows: u64, accesses: u64) -> Result<Self, ModelError> {
        if rows < TILE_WIDTH || !rows.is_power_of_two() || accesses > rows {
            return Err(ModelError::InvalidGeometry);
        }
        let mut previous = accesses;
        for (index, (&live, &groups)) in self
            .live_blocks
            .iter()
            .zip(self.occupied_outer_groups.iter())
            .enumerate()
        {
            let block_width = 1_u64 << (index + 1);
            let domain = rows / block_width;
            let minimum = accesses.div_ceil(block_width);
            if live < minimum
                || live > previous
                || live > domain
                || live < previous.div_ceil(2)
                || groups > live
                || groups > ADDRESS_DOMAIN
                || (accesses != 0 && groups == 0)
            {
                return Err(ModelError::InvalidCensus);
            }
            previous = live;
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProducerAccounting {
    pub dense_address_plane_bytes: u64,
    pub low_major_claim_view_bytes: u64,
    pub microtile_view_bytes: u64,
    pub replaced_high_major_view_bytes: u64,
    pub co_materialized_output_bytes: u64,
    pub co_materialized_floor_ns: u64,
    pub late_conversion_bytes: u64,
    pub late_conversion_floor_ns: u64,
    pub retained_stage5_to_stage6b_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Projection {
    pub rows: u64,
    pub accesses: u64,
    pub last_metal_message: usize,
    pub prefix_products: u64,
    pub dense_products: u64,
    pub total_products: u64,
    pub perfect_bytes: u64,
    pub logical_bytes: u64,
    pub requested_bytes: u64,
    pub compute_floor_ns: u64,
    pub perfect_traffic_floor_ns: u64,
    pub logical_traffic_floor_ns: u64,
    pub requested_traffic_floor_ns: u64,
    pub active_floor_ns: u64,
    pub eighty_percent_active_bar_ns: u64,
    pub launch_floor_ns: u64,
    pub resident_incremental_pursuit_ns: u64,
    pub family_charged_pursuit_ns: u64,
    pub sequence_owned_bytes: u64,
    pub resident_bytes: u64,
    pub cutoff_readback_bytes: u64,
    pub producer: ProducerAccounting,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelError {
    InvalidGeometry,
    InvalidCensus,
    InvalidCutoff,
    InvalidRate,
    Overflow,
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGeometry => f.write_str("invalid RAM-RA successor geometry"),
            Self::InvalidCensus => f.write_str("invalid RAM-RA prefix support census"),
            Self::InvalidCutoff => f.write_str("invalid RAM-RA Metal/CPU cutoff"),
            Self::InvalidRate => f.write_str("RAM-RA roof rate must be nonzero"),
            Self::Overflow => f.write_str("RAM-RA analytical arithmetic overflowed"),
        }
    }
}

pub fn target_projection() -> Result<Projection, ModelError> {
    project(
        TARGET_ROWS,
        TARGET_ACCESSES,
        PrefixCensus::target_worst_case(),
        SELECTED_LAST_METAL_MESSAGE,
        CPU_CONTINUATION_AFTER_MESSAGE_16_NS,
    )
}

pub fn project(
    rows: u64,
    accesses: u64,
    census: PrefixCensus,
    last_metal_message: usize,
    cpu_continuation_ns: u64,
) -> Result<Projection, ModelError> {
    if rows < TILE_WIDTH
        || !rows.is_power_of_two()
        || !rows.is_multiple_of(TILE_WIDTH)
        || accesses > rows
    {
        return Err(ModelError::InvalidGeometry);
    }
    let log_t = rows.ilog2() as usize;
    if last_metal_message < PREFIX_ROUNDS || last_metal_message >= log_t {
        return Err(ModelError::InvalidCutoff);
    }
    let census = census.validate(rows, accesses)?;
    let tiles = rows / TILE_WIDTH;

    let branch_products = checked_mul(2, checked_mul(FACTORS * BINS, 1 + 2 + 4 + 8)?)?;
    let mut prefix_products = branch_products;
    for (&live, &groups) in census
        .live_blocks
        .iter()
        .zip(census.occupied_outer_groups.iter())
    {
        prefix_products = checked_add(prefix_products, checked_mul(4, live)?)?;
        prefix_products = checked_add(prefix_products, checked_mul(2, groups)?)?;
    }

    let mut dense_products = 0;
    for round in PREFIX_ROUNDS..=last_metal_message {
        let source = checked_mul(2, rows >> round)?;
        let (_, e_out) = weight_shape(log_t, round)?;
        dense_products = checked_add(dense_products, checked_mul(2, source)?)?;
        dense_products = checked_add(dense_products, checked_mul(2, e_out)?)?;
    }
    let total_products = checked_add(prefix_products, dense_products)?;

    let (e_in_fields, e_out_fields) = weight_pyramid_fields(log_t, last_metal_message)?;
    let weight_upload = checked_mul(checked_add(e_in_fields, e_out_fields)?, FIELD_BYTES)?;
    let initial_chunk_tables = checked_mul(FACTORS * BINS, FIELD_BYTES)?;
    let mut perfect_bytes = checked_add(initial_chunk_tables, weight_upload)?;
    let mut logical_bytes = perfect_bytes;
    let mut requested_bytes = perfect_bytes;
    let mut branch_unique_total = 0;
    let mut branch_requested_total = 0;

    for round in 0..PREFIX_ROUNDS {
        let unique_view = checked_sum(&[
            checked_mul(MASK_BYTES, tiles)?,
            checked_mul(OFFSET_BYTES, checked_add(tiles, 1)?)?,
            checked_mul(ADDRESS_BYTES, accesses)?,
        ])?;
        let logical_view = checked_sum(&[
            checked_mul(MASK_BYTES, tiles)?,
            checked_mul(2 * OFFSET_BYTES, tiles)?,
            checked_mul(ADDRESS_BYTES, accesses)?,
        ])?;
        let branch_unique = checked_mul(
            checked_mul(FACTORS * BINS * FIELD_BYTES, 1_u64 << round)?,
            1,
        )?;
        let branch_requested = checked_mul(FACTORS * FIELD_BYTES, accesses)?;
        let (e_in, e_out) = weight_shape(log_t, round)?;
        let common = checked_add(
            checked_mul(checked_add(e_in, e_out)?, FIELD_BYTES)?,
            reduction_bytes(e_out)?,
        )?;

        perfect_bytes = checked_add(
            perfect_bytes,
            checked_sum(&[unique_view, branch_unique, common])?,
        )?;
        logical_bytes = checked_add(
            logical_bytes,
            checked_sum(&[logical_view, branch_unique, common])?,
        )?;
        requested_bytes = checked_add(
            requested_bytes,
            checked_sum(&[logical_view, branch_requested, common])?,
        )?;
        branch_unique_total = checked_add(branch_unique_total, branch_unique)?;
        branch_requested_total = checked_add(branch_requested_total, branch_requested)?;

        if round + 1 == PREFIX_ROUNDS {
            let materialized_state = checked_mul(FACTORS * FIELD_BYTES, tiles)?;
            perfect_bytes = checked_add(perfect_bytes, materialized_state)?;
            logical_bytes = checked_add(logical_bytes, materialized_state)?;
            requested_bytes = checked_add(requested_bytes, materialized_state)?;
        }
    }

    for width in [1_u64, 2, 4, 8] {
        let source_fields = checked_mul(FACTORS * BINS, width)?;
        let traffic = checked_mul(48, source_fields)?;
        perfect_bytes = checked_add(perfect_bytes, traffic)?;
        logical_bytes = checked_add(logical_bytes, traffic)?;
        requested_bytes = checked_add(requested_bytes, traffic)?;
    }

    for round in PREFIX_ROUNDS..=last_metal_message {
        let source = checked_mul(2, rows >> round)?;
        let (e_in, e_out) = weight_shape(log_t, round)?;
        let traffic = checked_sum(&[
            checked_mul(48, source)?,
            checked_mul(checked_add(e_in, e_out)?, FIELD_BYTES)?,
            reduction_bytes(e_out)?,
        ])?;
        perfect_bytes = checked_add(perfect_bytes, traffic)?;
        logical_bytes = checked_add(logical_bytes, traffic)?;
        requested_bytes = checked_add(requested_bytes, traffic)?;
    }

    let metal_messages = (last_metal_message + 1) as u64;
    let scalar_readback = checked_mul(metal_messages, MESSAGE_COLUMNS * FIELD_BYTES)?;
    let cutoff_elements = rows >> last_metal_message;
    let cutoff_readback_bytes = checked_mul(FACTORS * FIELD_BYTES, cutoff_elements)?;
    for total in [&mut perfect_bytes, &mut logical_bytes, &mut requested_bytes] {
        *total = checked_add(*total, scalar_readback)?;
        *total = checked_add(*total, cutoff_readback_bytes)?;
    }

    debug_assert_eq!(branch_unique_total, 253_952);
    debug_assert_eq!(branch_requested_total, 160 * accesses);

    let compute_floor_ns = rate_ns(total_products, RETAINED_FIELD_PRODUCTS_PER_SECOND)?;
    let perfect_traffic_floor_ns = rate_ns(perfect_bytes, RETAINED_COPY_BYTES_PER_SECOND)?;
    let logical_traffic_floor_ns = rate_ns(logical_bytes, RETAINED_COPY_BYTES_PER_SECOND)?;
    let requested_traffic_floor_ns = rate_ns(requested_bytes, RETAINED_COPY_BYTES_PER_SECOND)?;
    let active_floor_ns = compute_floor_ns.max(requested_traffic_floor_ns);
    let eighty_percent_active_bar_ns = div_ceil(
        checked_mul(active_floor_ns, 1_000)?,
        ROOF_EFFICIENCY_PERMILLE,
    );
    let launch_floor_ns = checked_mul(metal_messages, COMMAND_WALL_FLOOR_NS)?;

    let producer = producer_accounting(rows, accesses)?;
    let resident_incremental_pursuit_ns = checked_sum(&[
        eighty_percent_active_bar_ns,
        launch_floor_ns,
        SEQUENCE_SETUP_FLOOR_NS,
        cpu_continuation_ns,
    ])?;
    let family_charged_pursuit_ns = checked_add(
        resident_incremental_pursuit_ns,
        producer.co_materialized_floor_ns,
    )?;

    let branch_fields = checked_mul(FACTORS * BINS, 8 + 16)?;
    let dense_fields = checked_mul(FACTORS, checked_add(tiles, tiles / 2)?)?;
    let partial_fields = checked_mul(2, MESSAGE_COLUMNS * max_e_out(log_t, last_metal_message)?)?;
    let sequence_owned_bytes = checked_mul(
        checked_sum(&[
            branch_fields,
            dense_fields,
            e_in_fields,
            e_out_fields,
            partial_fields,
        ])?,
        FIELD_BYTES,
    )?;
    let resident_bytes = checked_add(sequence_owned_bytes, producer.microtile_view_bytes)?;

    Ok(Projection {
        rows,
        accesses,
        last_metal_message,
        prefix_products,
        dense_products,
        total_products,
        perfect_bytes,
        logical_bytes,
        requested_bytes,
        compute_floor_ns,
        perfect_traffic_floor_ns,
        logical_traffic_floor_ns,
        requested_traffic_floor_ns,
        active_floor_ns,
        eighty_percent_active_bar_ns,
        launch_floor_ns,
        resident_incremental_pursuit_ns,
        family_charged_pursuit_ns,
        sequence_owned_bytes,
        resident_bytes,
        cutoff_readback_bytes,
        producer,
    })
}

pub fn producer_accounting(rows: u64, accesses: u64) -> Result<ProducerAccounting, ModelError> {
    if rows < TILE_WIDTH || !rows.is_power_of_two() || accesses > rows {
        return Err(ModelError::InvalidGeometry);
    }
    let tiles = rows / TILE_WIDTH;
    let dense_address_plane_bytes = checked_mul(rows, 4)?;
    let low_major_claim_view_bytes = checked_add(
        checked_mul(accesses, 4)?,
        checked_mul(ADDRESS_DOMAIN + 1, OFFSET_BYTES)?,
    )?;
    let replaced_high_major_view_bytes = low_major_claim_view_bytes;
    let microtile_view_bytes = checked_sum(&[
        checked_mul(tiles, MASK_BYTES)?,
        checked_mul(checked_add(tiles, 1)?, OFFSET_BYTES)?,
        checked_mul(accesses, ADDRESS_BYTES)?,
    ])?;
    let co_materialized_output_bytes = checked_sum(&[
        dense_address_plane_bytes,
        low_major_claim_view_bytes,
        microtile_view_bytes,
    ])?;
    let late_conversion_bytes = checked_add(dense_address_plane_bytes, microtile_view_bytes)?;
    Ok(ProducerAccounting {
        dense_address_plane_bytes,
        low_major_claim_view_bytes,
        microtile_view_bytes,
        replaced_high_major_view_bytes,
        co_materialized_output_bytes,
        co_materialized_floor_ns: rate_ns(
            co_materialized_output_bytes,
            RETAINED_COPY_BYTES_PER_SECOND,
        )?,
        late_conversion_bytes,
        late_conversion_floor_ns: rate_ns(late_conversion_bytes, RETAINED_COPY_BYTES_PER_SECOND)?,
        retained_stage5_to_stage6b_bytes: microtile_view_bytes,
    })
}

fn weight_shape(log_t: usize, round: usize) -> Result<(u64, u64), ModelError> {
    if round >= log_t {
        return Err(ModelError::InvalidCutoff);
    }
    let split = log_t / 2;
    let head_bits = log_t - round - 1;
    let e_out_bits = head_bits.min(split);
    let e_in_bits = head_bits - e_out_bits;
    Ok((1_u64 << e_in_bits, 1_u64 << e_out_bits))
}

fn weight_pyramid_fields(
    log_t: usize,
    last_metal_message: usize,
) -> Result<(u64, u64), ModelError> {
    let mut e_in_fields = 0;
    let mut e_out_fields = 0;
    let mut previous_e_in = None;
    let mut previous_e_out = None;
    for round in 0..=last_metal_message {
        let (e_in, e_out) = weight_shape(log_t, round)?;
        if previous_e_in != Some(e_in) {
            e_in_fields = checked_add(e_in_fields, e_in)?;
            previous_e_in = Some(e_in);
        }
        if previous_e_out != Some(e_out) {
            e_out_fields = checked_add(e_out_fields, e_out)?;
            previous_e_out = Some(e_out);
        }
    }
    Ok((e_in_fields, e_out_fields))
}

fn max_e_out(log_t: usize, last_metal_message: usize) -> Result<u64, ModelError> {
    (0..=last_metal_message).try_fold(0, |maximum, round| {
        weight_shape(log_t, round).map(|(_, e_out)| maximum.max(e_out))
    })
}

fn reduction_bytes(mut input: u64) -> Result<u64, ModelError> {
    let mut total = checked_mul(MESSAGE_COLUMNS * FIELD_BYTES, input)?;
    while input > 1 {
        let output = input.div_ceil(SIMD_WIDTH);
        total = checked_add(
            total,
            checked_mul(MESSAGE_COLUMNS * FIELD_BYTES, checked_add(input, output)?)?,
        )?;
        input = output;
    }
    Ok(total)
}

fn rate_ns(work: u64, rate: u64) -> Result<u64, ModelError> {
    if rate == 0 {
        return Err(ModelError::InvalidRate);
    }
    let result = (u128::from(work) * NANOS_PER_SECOND).div_ceil(u128::from(rate));
    u64::try_from(result).map_err(|_| ModelError::Overflow)
}

fn checked_sum(values: &[u64]) -> Result<u64, ModelError> {
    values
        .iter()
        .try_fold(0, |sum, value| checked_add(sum, *value))
}

fn checked_add(left: u64, right: u64) -> Result<u64, ModelError> {
    left.checked_add(right).ok_or(ModelError::Overflow)
}

fn checked_mul(left: u64, right: u64) -> Result<u64, ModelError> {
    left.checked_mul(right).ok_or(ModelError::Overflow)
}

const fn div_ceil(value: u64, divisor: u64) -> u64 {
    value.div_ceil(divisor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_counts_are_frozen() {
        let projection = target_projection().unwrap();
        assert_eq!(projection.prefix_products, 213_926_400);
        assert_eq!(projection.dense_products, 16_919_552);
        assert_eq!(projection.total_products, 230_845_952);
        assert_eq!(projection.perfect_bytes, 893_151_988);
        assert_eq!(projection.logical_bytes, 977_038_048);
        assert_eq!(projection.requested_bytes, 4_496_784_096);
        assert_eq!(projection.compute_floor_ns, 12_753_920);
        assert_eq!(projection.perfect_traffic_floor_ns, 1_977_305);
        assert_eq!(projection.logical_traffic_floor_ns, 2_163_017);
        assert_eq!(projection.requested_traffic_floor_ns, 9_955_208);
        assert_eq!(projection.active_floor_ns, 12_753_920);
        assert_eq!(projection.eighty_percent_active_bar_ns, 15_942_400);
        assert_eq!(projection.launch_floor_ns, 2_397_000);
        assert_eq!(projection.resident_incremental_pursuit_ns, 19_228_606);
        assert_eq!(projection.family_charged_pursuit_ns, 20_170_897);
        assert_eq!(projection.sequence_owned_bytes, 202_432_496);
        assert_eq!(projection.resident_bytes, 271_598_324);
        assert_eq!(projection.cutoff_readback_bytes, 32_768);
    }

    #[test]
    fn producer_is_priced_once() {
        let producer = producer_accounting(TARGET_ROWS, TARGET_ACCESSES).unwrap();
        assert_eq!(producer.dense_address_plane_bytes, 268_435_456);
        assert_eq!(producer.low_major_claim_view_bytes, 88_032_772);
        assert_eq!(producer.replaced_high_major_view_bytes, 88_032_772);
        assert_eq!(producer.microtile_view_bytes, 69_165_828);
        assert_eq!(producer.co_materialized_output_bytes, 425_634_056);
        assert_eq!(producer.co_materialized_floor_ns, 942_291);
        assert_eq!(producer.late_conversion_bytes, 337_601_284);
        assert_eq!(producer.late_conversion_floor_ns, 747_399);
    }

    #[test]
    fn target_caps_use_the_complete_member_median() {
        assert_eq!(FIVE_X_CAP_NS, 54_933_158);
        assert_eq!(EIGHT_X_CAP_NS, 34_333_223);
        assert!(COMPLETE_PURSUIT_BAR_NS < EIGHT_X_CAP_NS);
    }

    #[test]
    fn malformed_census_is_rejected() {
        let mut census = PrefixCensus::target_worst_case();
        census.live_blocks[1] = census.live_blocks[0] + 1;
        assert_eq!(
            project(
                TARGET_ROWS,
                TARGET_ACCESSES,
                census,
                SELECTED_LAST_METAL_MESSAGE,
                CPU_CONTINUATION_AFTER_MESSAGE_16_NS,
            ),
            Err(ModelError::InvalidCensus)
        );
    }
}
