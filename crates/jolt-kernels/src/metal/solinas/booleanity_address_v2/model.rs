//! Pre-registered log-27 traffic, operation, cutoff, and campaign model.

use core::cmp::Ordering;

use super::{
    checked_mul, checked_sum, BooleanityAddressV2BufferLengths, BooleanityAddressV2Error,
    BooleanityAddressV2Geometry, BOOLEANITY_ADDRESS_V2_ACCUMULATOR_THREADS,
    BOOLEANITY_ADDRESS_V2_BINS, BOOLEANITY_ADDRESS_V2_FIELD_BYTES,
    BOOLEANITY_ADDRESS_V2_FINALIZE_THREADGROUP_BYTES, BOOLEANITY_ADDRESS_V2_FINALIZE_THREADS,
    BOOLEANITY_ADDRESS_V2_FIRST_TILE_SELECTORS, BOOLEANITY_ADDRESS_V2_HOT_PLANES,
    BOOLEANITY_ADDRESS_V2_REMAINING_SELECTORS, BOOLEANITY_ADDRESS_V2_REMAINING_TILES,
    BOOLEANITY_ADDRESS_V2_ROW_BYTES, BOOLEANITY_ADDRESS_V2_SELECTORS,
    BOOLEANITY_ADDRESS_V2_TARGET_LOG_T, BOOLEANITY_ADDRESS_V2_THREADGROUP_BYTES,
};

pub const LOG_T_27: usize = 27;
pub const ROWS_AT_LOG_T_27: u64 = 1 << LOG_T_27;
pub const RETAINED_INNER_LOG2: usize = 15;
pub const COPY_GIB_PER_SECOND: f64 = 420.68;
pub const FIELD_PRODUCTS_PER_SECOND: f64 = 16.42e9;
pub const RETAINED_LOG27_SPEEDUP: f64 = 4.969_700_993;
pub const ROBUST_SPEEDUP_NUMERATOR: u64 = 53;
pub const ROBUST_SPEEDUP_DENOMINATOR: u64 = 10;
pub const PAIR_FLOOR_NUMERATOR: u64 = 5;
pub const PAIR_FLOOR_DENOMINATOR: u64 = 1;
pub const RETAINED_IMPROVEMENT_NUMERATOR: u64 = 103;
pub const RETAINED_IMPROVEMENT_DENOMINATOR: u64 = 100;

const EQUALITY_PASSES: u64 = 1 + BOOLEANITY_ADDRESS_V2_REMAINING_TILES as u64;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TrafficModel {
    pub resident_row_read_bytes: u64,
    pub projection_write_bytes: u64,
    pub packed_read_bytes: u64,
    pub e_in_cache_unique_bytes: u64,
    pub e_out_cache_unique_bytes: u64,
    pub e_in_issued_bytes: u64,
    pub e_out_issued_bytes: u64,
    pub partial_write_read_bytes: u64,
    pub output_write_read_bytes: u64,
    pub first_phase_cache_optimistic_bytes: u64,
    pub packed_phase_cache_optimistic_bytes: u64,
    pub finalize_phase_cache_optimistic_bytes: u64,
    pub cache_optimistic_bytes: u64,
    pub fully_issued_bytes: u64,
    pub owned_bytes: u64,
    pub bucket_products: u64,
}

impl TrafficModel {
    pub fn retained_log27() -> Result<Self, BooleanityAddressV2Error> {
        Self::new(ROWS_AT_LOG_T_27, RETAINED_INNER_LOG2, 30, 25)
    }

    pub fn candidate(
        geometry: BooleanityAddressV2Geometry,
    ) -> Result<Self, BooleanityAddressV2Error> {
        Self::new(
            geometry.rows() as u64,
            geometry.inner_log2(),
            BOOLEANITY_ADDRESS_V2_HOT_PLANES as u64,
            BOOLEANITY_ADDRESS_V2_REMAINING_SELECTORS as u64,
        )
    }

    fn new(
        rows: u64,
        inner_log2: usize,
        projection_bytes_per_row: u64,
        packed_read_bytes_per_row: u64,
    ) -> Result<Self, BooleanityAddressV2Error> {
        let e_in = 1u64 << inner_log2;
        let e_out = rows / e_in;
        let output_fields = checked_mul(
            BOOLEANITY_ADDRESS_V2_SELECTORS as u64,
            BOOLEANITY_ADDRESS_V2_BINS as u64,
        )?;
        let partial_fields = checked_mul(output_fields, e_out)?;
        let first_partial_fields = checked_mul(
            checked_mul(
                BOOLEANITY_ADDRESS_V2_FIRST_TILE_SELECTORS as u64,
                BOOLEANITY_ADDRESS_V2_BINS as u64,
            )?,
            e_out,
        )?;
        let remaining_partial_fields = partial_fields
            .checked_sub(first_partial_fields)
            .ok_or(BooleanityAddressV2Error::ArithmeticOverflow)?;

        let resident_row_read_bytes = checked_mul(BOOLEANITY_ADDRESS_V2_ROW_BYTES, rows)?;
        let projection_write_bytes = checked_mul(projection_bytes_per_row, rows)?;
        let packed_read_bytes = checked_mul(packed_read_bytes_per_row, rows)?;
        let e_in_cache_unique_bytes = checked_mul(BOOLEANITY_ADDRESS_V2_FIELD_BYTES, e_in)?;
        let e_out_cache_unique_bytes = checked_mul(BOOLEANITY_ADDRESS_V2_FIELD_BYTES, e_out)?;
        let e_in_issued_bytes = checked_mul(
            checked_mul(BOOLEANITY_ADDRESS_V2_FIELD_BYTES, rows)?,
            EQUALITY_PASSES,
        )?;
        let e_out_issued_bytes = checked_mul(
            checked_mul(BOOLEANITY_ADDRESS_V2_FIELD_BYTES, e_out)?,
            EQUALITY_PASSES,
        )?;
        let partial_write_read_bytes = checked_mul(
            checked_mul(2 * BOOLEANITY_ADDRESS_V2_FIELD_BYTES, partial_fields)?,
            1,
        )?;
        let output_write_read_bytes =
            checked_mul(2 * BOOLEANITY_ADDRESS_V2_FIELD_BYTES, output_fields)?;
        let first_partial_write_bytes =
            checked_mul(BOOLEANITY_ADDRESS_V2_FIELD_BYTES, first_partial_fields)?;
        let remaining_partial_write_bytes =
            checked_mul(BOOLEANITY_ADDRESS_V2_FIELD_BYTES, remaining_partial_fields)?;
        let partial_read_bytes = checked_mul(BOOLEANITY_ADDRESS_V2_FIELD_BYTES, partial_fields)?;

        let first_phase_cache_optimistic_bytes = checked_sum(&[
            resident_row_read_bytes,
            projection_write_bytes,
            e_in_cache_unique_bytes,
            e_out_cache_unique_bytes,
            first_partial_write_bytes,
        ])?;
        let packed_phase_cache_optimistic_bytes =
            checked_sum(&[packed_read_bytes, remaining_partial_write_bytes])?;
        let finalize_phase_cache_optimistic_bytes =
            checked_sum(&[partial_read_bytes, output_write_read_bytes])?;
        let cache_optimistic_bytes = checked_sum(&[
            first_phase_cache_optimistic_bytes,
            packed_phase_cache_optimistic_bytes,
            finalize_phase_cache_optimistic_bytes,
        ])?;
        let fully_issued_bytes = checked_sum(&[
            resident_row_read_bytes,
            projection_write_bytes,
            packed_read_bytes,
            e_in_issued_bytes,
            e_out_issued_bytes,
            partial_write_read_bytes,
            output_write_read_bytes,
        ])?;
        let owned_bytes = checked_sum(&[
            projection_write_bytes,
            e_in_cache_unique_bytes,
            e_out_cache_unique_bytes,
            partial_read_bytes,
            output_write_read_bytes / 2,
        ])?;

        Ok(Self {
            resident_row_read_bytes,
            projection_write_bytes,
            packed_read_bytes,
            e_in_cache_unique_bytes,
            e_out_cache_unique_bytes,
            e_in_issued_bytes,
            e_out_issued_bytes,
            partial_write_read_bytes,
            output_write_read_bytes,
            first_phase_cache_optimistic_bytes,
            packed_phase_cache_optimistic_bytes,
            finalize_phase_cache_optimistic_bytes,
            cache_optimistic_bytes,
            fully_issued_bytes,
            owned_bytes,
            bucket_products: partial_fields,
        })
    }

    pub fn cache_optimistic_copy_floor_ns(self) -> u64 {
        gib_floor_ns(self.cache_optimistic_bytes, COPY_GIB_PER_SECOND)
    }

    pub fn useful_products_per_cache_byte(self) -> f64 {
        self.bucket_products as f64 / self.cache_optimistic_bytes as f64
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkloadCensus {
    pub rows: u64,
    pub bytecode_present_rows: u64,
    pub ram_present_rows: u64,
    /// Rows whose selectors 24, 25, and 26 all target bucket zero.
    pub common_high_increment_rows: u64,
}

impl WorkloadCensus {
    pub fn validate(self) -> Result<Self, BooleanityAddressV2Error> {
        if self.rows == 0 {
            return Err(BooleanityAddressV2Error::InvalidCensus {
                name: "rows",
                rows: 0,
                got: 0,
            });
        }
        for (name, got) in [
            ("bytecode present", self.bytecode_present_rows),
            ("RAM present", self.ram_present_rows),
            ("common high increment", self.common_high_increment_rows),
        ] {
            if got > self.rows {
                return Err(BooleanityAddressV2Error::InvalidCensus {
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WorkModel {
    pub selector_row_opportunities: u64,
    pub present_field_contributions: u64,
    pub local_field_additions: u64,
    pub atomic_field_additions: u64,
    pub first_phase_atomic_field_additions: u64,
    pub packed_phase_atomic_field_additions: u64,
    pub bucket_products: u64,
    pub first_phase_bucket_products: u64,
    pub packed_phase_bucket_products: u64,
    pub bucket_product_floor_ns: u64,
}

impl WorkModel {
    pub fn candidate(
        census: WorkloadCensus,
        geometry: BooleanityAddressV2Geometry,
    ) -> Result<Self, BooleanityAddressV2Error> {
        let census = census.validate()?;
        if census.rows != geometry.rows() as u64 {
            return Err(BooleanityAddressV2Error::InvalidCensus {
                name: "geometry rows",
                rows: geometry.rows() as u64,
                got: census.rows,
            });
        }
        let selector_row_opportunities =
            checked_mul(BOOLEANITY_ADDRESS_V2_SELECTORS as u64, census.rows)?;
        let present_field_contributions = checked_sum(&[
            checked_mul(25, census.rows)?,
            checked_mul(2, census.bytecode_present_rows)?,
            checked_mul(2, census.ram_present_rows)?,
        ])?;
        let workers = checked_mul(
            BOOLEANITY_ADDRESS_V2_ACCUMULATOR_THREADS as u64,
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
            .ok_or(BooleanityAddressV2Error::ArithmeticOverflow)?;
        // The raw tile owns lookup selectors 0--1 plus both optional pairs.
        let first_phase_atomic_field_additions = checked_sum(&[
            checked_mul(2, census.rows)?,
            checked_mul(2, census.bytecode_present_rows)?,
            checked_mul(2, census.ram_present_rows)?,
        ])?;
        let packed_phase_atomic_field_additions = atomic_field_additions
            .checked_sub(first_phase_atomic_field_additions)
            .ok_or(BooleanityAddressV2Error::ArithmeticOverflow)?;
        let bucket_products = checked_mul(
            checked_mul(
                BOOLEANITY_ADDRESS_V2_SELECTORS as u64,
                BOOLEANITY_ADDRESS_V2_BINS as u64,
            )?,
            geometry.e_out_length() as u64,
        )?;
        let first_phase_bucket_products = checked_mul(
            checked_mul(
                BOOLEANITY_ADDRESS_V2_FIRST_TILE_SELECTORS as u64,
                BOOLEANITY_ADDRESS_V2_BINS as u64,
            )?,
            geometry.e_out_length() as u64,
        )?;
        let packed_phase_bucket_products = bucket_products
            .checked_sub(first_phase_bucket_products)
            .ok_or(BooleanityAddressV2Error::ArithmeticOverflow)?;

        Ok(Self {
            selector_row_opportunities,
            present_field_contributions,
            local_field_additions: checked_sum(&[census.common_high_increment_rows, census.rows])?,
            atomic_field_additions,
            first_phase_atomic_field_additions,
            packed_phase_atomic_field_additions,
            bucket_products,
            first_phase_bucket_products,
            packed_phase_bucket_products,
            bucket_product_floor_ns: rate_floor_ns(bucket_products, FIELD_PRODUCTS_PER_SECOND),
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PhaseRoof {
    pub traffic_floor_ns: u64,
    pub atomic_floor_ns: u64,
    pub product_floor_ns: u64,
    pub binding_floor_ns: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CalibratedRoof {
    pub first: PhaseRoof,
    pub packed: PhaseRoof,
    pub finalize: PhaseRoof,
    pub bottomed_out_ns: u64,
    pub eighty_percent_cap_ns: u64,
}

impl CalibratedRoof {
    pub fn new(
        traffic: TrafficModel,
        work: WorkModel,
        atomic_field_additions_per_second: f64,
    ) -> Option<Self> {
        if !atomic_field_additions_per_second.is_finite()
            || atomic_field_additions_per_second <= 0.0
        {
            return None;
        }
        let first = phase_roof(
            traffic.first_phase_cache_optimistic_bytes,
            work.first_phase_atomic_field_additions,
            work.first_phase_bucket_products,
            atomic_field_additions_per_second,
        );
        let packed = phase_roof(
            traffic.packed_phase_cache_optimistic_bytes,
            work.packed_phase_atomic_field_additions,
            work.packed_phase_bucket_products,
            atomic_field_additions_per_second,
        );
        let finalize = PhaseRoof {
            traffic_floor_ns: gib_floor_ns(
                traffic.finalize_phase_cache_optimistic_bytes,
                COPY_GIB_PER_SECOND,
            ),
            atomic_floor_ns: 0,
            product_floor_ns: 0,
            binding_floor_ns: gib_floor_ns(
                traffic.finalize_phase_cache_optimistic_bytes,
                COPY_GIB_PER_SECOND,
            ),
        };
        let bottomed_out_ns = first
            .binding_floor_ns
            .saturating_add(packed.binding_floor_ns)
            .saturating_add(finalize.binding_floor_ns);
        Some(Self {
            first,
            packed,
            finalize,
            bottomed_out_ns,
            eighty_percent_cap_ns: bottomed_out_ns.saturating_mul(5) / 4,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PipelineAdmission {
    pub max_buffer_bytes: u64,
    pub available_working_set_bytes: u64,
    pub accumulator_max_threads: usize,
    pub finalize_max_threads: usize,
    pub max_threadgroup_bytes: usize,
    pub accumulator_private_bytes: u64,
    pub accumulator_spills: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CutoffDecision {
    RetainLog26,
    Reject(&'static str),
    ScreenV2,
}

pub fn cutoff(
    geometry: BooleanityAddressV2Geometry,
    admission: PipelineAdmission,
) -> Result<CutoffDecision, BooleanityAddressV2Error> {
    if geometry.log_t() < BOOLEANITY_ADDRESS_V2_TARGET_LOG_T {
        return Ok(CutoffDecision::RetainLog26);
    }
    let lengths = geometry.buffer_lengths()?;
    let buffers = [
        lengths.hot_bytes,
        checked_mul(lengths.e_in_fields, BOOLEANITY_ADDRESS_V2_FIELD_BYTES)?,
        checked_mul(lengths.e_out_fields, BOOLEANITY_ADDRESS_V2_FIELD_BYTES)?,
        checked_mul(lengths.partial_fields, BOOLEANITY_ADDRESS_V2_FIELD_BYTES)?,
        checked_mul(lengths.output_fields, BOOLEANITY_ADDRESS_V2_FIELD_BYTES)?,
    ];
    if buffers
        .into_iter()
        .any(|bytes| bytes > admission.max_buffer_bytes)
    {
        return Ok(CutoffDecision::Reject("buffer length"));
    }
    if lengths.owned_bytes()? > admission.available_working_set_bytes {
        return Ok(CutoffDecision::Reject("working set"));
    }
    if admission.accumulator_max_threads < BOOLEANITY_ADDRESS_V2_ACCUMULATOR_THREADS {
        return Ok(CutoffDecision::Reject("accumulator threads"));
    }
    if admission.finalize_max_threads < BOOLEANITY_ADDRESS_V2_FINALIZE_THREADS {
        return Ok(CutoffDecision::Reject("finalizer threads"));
    }
    if admission.max_threadgroup_bytes < BOOLEANITY_ADDRESS_V2_THREADGROUP_BYTES
        || admission.max_threadgroup_bytes < BOOLEANITY_ADDRESS_V2_FINALIZE_THREADGROUP_BYTES
    {
        return Ok(CutoffDecision::Reject("threadgroup memory"));
    }
    if admission.accumulator_spills || admission.accumulator_private_bytes != 0 {
        return Ok(CutoffDecision::Reject("accumulator spill/private memory"));
    }
    Ok(CutoffDecision::ScreenV2)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RationalSpeedup {
    pub cpu_ns: u64,
    pub metal_ns: u64,
}

impl RationalSpeedup {
    pub const fn new(cpu_ns: u64, metal_ns: u64) -> Self {
        Self { cpu_ns, metal_ns }
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
pub struct ScreenSample {
    pub cpu_member_ns: u64,
    pub retained_member_ns: u64,
    pub candidate_member_ns: u64,
    pub masses_exact: bool,
    pub transcript_exact: bool,
    pub proof_verified: bool,
    pub receipt_exact: bool,
    pub hamming_not_slower: bool,
    pub family_not_slower: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScreenDecision {
    Kill(&'static str),
    RunCampaign,
}

pub fn evaluate_screen(sample: ScreenSample) -> ScreenDecision {
    if !sample.masses_exact
        || !sample.transcript_exact
        || !sample.proof_verified
        || !sample.receipt_exact
    {
        return ScreenDecision::Kill("parity or receipt");
    }
    if !sample.hamming_not_slower || !sample.family_not_slower {
        return ScreenDecision::Kill("downstream family regression");
    }
    if !RationalSpeedup::new(sample.cpu_member_ns, sample.candidate_member_ns)
        .clears(ROBUST_SPEEDUP_NUMERATOR, ROBUST_SPEEDUP_DENOMINATOR)
    {
        return ScreenDecision::Kill("5.3x complete-member bar");
    }
    if !RationalSpeedup::new(sample.retained_member_ns, sample.candidate_member_ns).clears(
        RETAINED_IMPROVEMENT_NUMERATOR,
        RETAINED_IMPROVEMENT_DENOMINATOR,
    ) {
        return ScreenDecision::Kill("3% retained-arm improvement");
    }
    ScreenDecision::RunCampaign
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CampaignOrder {
    CpuFirst,
    V2First,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PairSample {
    pub order: CampaignOrder,
    pub cpu_member_ns: u64,
    pub candidate_member_ns: u64,
    pub evidence_exact: bool,
}

impl PairSample {
    pub const fn speedup(self) -> RationalSpeedup {
        RationalSpeedup::new(self.cpu_member_ns, self.candidate_member_ns)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CampaignSummary {
    pub minimum_speedup: f64,
    pub median_speedup: f64,
    pub cpu_first_median_speedup: f64,
    pub v2_first_median_speedup: f64,
    pub sealed_holdout_speedup: f64,
}

pub fn evaluate_campaign(
    pairs: &[PairSample],
    sealed_holdout: PairSample,
) -> Result<Option<CampaignSummary>, BooleanityAddressV2Error> {
    if pairs.len() != 5 {
        return Err(BooleanityAddressV2Error::CampaignSize(pairs.len()));
    }
    for (index, pair) in pairs.iter().enumerate() {
        let expected = if index % 2 == 0 {
            CampaignOrder::CpuFirst
        } else {
            CampaignOrder::V2First
        };
        if pair.order != expected {
            return Err(BooleanityAddressV2Error::CampaignOrder(index));
        }
        if !pair.evidence_exact {
            return Err(BooleanityAddressV2Error::CampaignEvidence(index));
        }
        if !pair
            .speedup()
            .clears(PAIR_FLOOR_NUMERATOR, PAIR_FLOOR_DENOMINATOR)
        {
            return Ok(None);
        }
    }
    if !sealed_holdout.evidence_exact {
        return Err(BooleanityAddressV2Error::CampaignEvidence(5));
    }
    let mut all = pairs
        .iter()
        .map(|pair| pair.speedup().as_f64())
        .collect::<Vec<_>>();
    let mut cpu_first = pairs
        .iter()
        .filter(|pair| pair.order == CampaignOrder::CpuFirst)
        .map(|pair| pair.speedup().as_f64())
        .collect::<Vec<_>>();
    let mut v2_first = pairs
        .iter()
        .filter(|pair| pair.order == CampaignOrder::V2First)
        .map(|pair| pair.speedup().as_f64())
        .collect::<Vec<_>>();
    let minimum_speedup = all.iter().copied().fold(f64::INFINITY, f64::min);
    let median_speedup = median(&mut all);
    let cpu_first_median_speedup = median(&mut cpu_first);
    let v2_first_median_speedup = median(&mut v2_first);
    let sealed_holdout_speedup = sealed_holdout.speedup().as_f64();
    let robust = ROBUST_SPEEDUP_NUMERATOR as f64 / ROBUST_SPEEDUP_DENOMINATOR as f64;
    if median_speedup < robust
        || cpu_first_median_speedup < robust
        || v2_first_median_speedup < robust
        || sealed_holdout_speedup < robust
    {
        return Ok(None);
    }
    Ok(Some(CampaignSummary {
        minimum_speedup,
        median_speedup,
        cpu_first_median_speedup,
        v2_first_median_speedup,
        sealed_holdout_speedup,
    }))
}

pub fn log27_traffic_projection() -> Result<f64, BooleanityAddressV2Error> {
    let retained = TrafficModel::retained_log27()?;
    let candidate =
        TrafficModel::candidate(BooleanityAddressV2Geometry::new(ROWS_AT_LOG_T_27 as usize)?)?;
    Ok(
        RETAINED_LOG27_SPEEDUP * retained.cache_optimistic_bytes as f64
            / candidate.cache_optimistic_bytes as f64,
    )
}

pub fn required_log27_throughput_gain() -> f64 {
    (ROBUST_SPEEDUP_NUMERATOR as f64 / ROBUST_SPEEDUP_DENOMINATOR as f64) / RETAINED_LOG27_SPEEDUP
}

fn phase_roof(
    traffic_bytes: u64,
    atomic_additions: u64,
    products: u64,
    atomic_additions_per_second: f64,
) -> PhaseRoof {
    let traffic_floor_ns = gib_floor_ns(traffic_bytes, COPY_GIB_PER_SECOND);
    let atomic_floor_ns = rate_floor_ns(atomic_additions, atomic_additions_per_second);
    let product_floor_ns = rate_floor_ns(products, FIELD_PRODUCTS_PER_SECOND);
    PhaseRoof {
        traffic_floor_ns,
        atomic_floor_ns,
        product_floor_ns,
        binding_floor_ns: traffic_floor_ns.max(atomic_floor_ns).max(product_floor_ns),
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        f64::midpoint(values[middle - 1], values[middle])
    } else {
        values[middle]
    }
}

fn gib_floor_ns(bytes: u64, gib_per_second: f64) -> u64 {
    ((bytes as f64 * 1e9) / (gib_per_second * (1u64 << 30) as f64)).ceil() as u64
}

fn rate_floor_ns(operations: u64, operations_per_second: f64) -> u64 {
    ((operations as f64 * 1e9) / operations_per_second).ceil() as u64
}

pub fn expected_candidate_lengths_at_log27(
) -> Result<BooleanityAddressV2BufferLengths, BooleanityAddressV2Error> {
    BooleanityAddressV2Geometry::new(ROWS_AT_LOG_T_27 as usize)?.buffer_lengths()
}
