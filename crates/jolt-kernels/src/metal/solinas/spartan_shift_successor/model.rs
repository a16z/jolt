//! Exact work counts and fail-closed performance gates.

use super::abi::{
    SpartanShiftSuccessorAbiError, SpartanShiftSuccessorGeometry, FIELD_BYTES, FLAG_PLANES,
    MIDPOINT_FULL_TABLES, MIDPOINT_RESIDUAL_TABLES, OUTER_COMPONENT_TABLES, PREFIX_Q_TABLES,
    PRODUCT_COMPONENT_TABLES,
};

pub const TARGET_LOG_T: u32 = 26;
pub const TARGET_ROWS: usize = 1 << TARGET_LOG_T;

pub const FROZEN_CPU_ARTIFACT: &str = "benchmark-runs/metal-piop-eval/20260806-133709-697013";
pub const FROZEN_CPU_REVISION: &str = "5f520c21e338632aa0bf5936ceb02be6c22fa40f";
pub const FROZEN_CPU_SAMPLES_NS: [u64; 5] = [
    131_051_624,
    131_584_500,
    129_304_918,
    130_343_291,
    134_289_502,
];
pub const FROZEN_CPU_MEDIAN_NS: u64 = 131_051_624;
pub const FIVE_X_CAP_NS: u64 = FROZEN_CPU_MEDIAN_NS / 5;
pub const EIGHT_X_CAP_NS: u64 = FROZEN_CPU_MEDIAN_NS / 8;

pub const RETAINED_COPY_BYTES_PER_SECOND: u64 = 451_701_710_520;
pub const RETAINED_MATCHED_HALF_TERMS_PER_SECOND: u64 = 33_168_000_000;
pub const HALF_WIDTH_PROMOTION_FLOOR_PER_SECOND: u64 = 26_272_000_000;
pub const RETAINED_REGISTER_PRODUCTS_PER_SECOND: u64 = 18_100_000_000;
pub const ROOF_EFFICIENCY_PERMILLE: u64 = 800;
pub const INSTRUCTION_INPUT_ROW_BYTES: u64 = 48;
pub const SPARTAN_OUTER_RESIDUAL_ROW_BYTES: u64 = 112;

const NANOS_PER_SECOND: u128 = 1_000_000_000;
const PERMILLE: u128 = 1_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MidpointPlan {
    BorrowInstructionInputUpc,
    SelfContained,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttributionBoundary {
    GrossCoMaterializedCompact,
    GrossFreshFusedOuter,
    GrossFreshSplitOuter,
    ResidentPiopIncremental,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhaseWork {
    pub name: &'static str,
    pub half_width_terms: u128,
    pub full_products: u128,
    pub selected_field_adds_max: u128,
    pub logical_bytes: u128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkPlan {
    pub geometry: SpartanShiftSuccessorGeometry,
    pub midpoint: MidpointPlan,
    pub high_tiles: usize,
    pub outer_carrier: PhaseWork,
    pub product_carrier: PhaseWork,
    pub q_combine: PhaseWork,
    pub prefix_host: PhaseWork,
    pub midpoint_fold: PhaseWork,
    pub suffix_host: PhaseWork,
    pub outer_logical_source_bytes: u128,
    pub transient_upc_projection_bytes: u128,
    pub producer_projection_bytes: u128,
    pub host_handoff_bytes: u128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoofBounds {
    pub compute_floor_ns: u64,
    pub traffic_floor_ns: u64,
    pub binding_floor_ns: u64,
    pub eighty_percent_bar_ns: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComponentBars {
    pub projection_wall_ns: u64,
    pub outer_carrier_active_ns: u64,
    pub product_carrier_wall_ns: u64,
    pub midpoint_active_ns: u64,
    pub host_work_ns: u64,
    pub complete_five_x_ns: u64,
    pub complete_eight_x_ns: u64,
}

pub const TARGET_COMPONENT_BARS: ComponentBars = ComponentBars {
    projection_wall_ns: 1_000_000,
    outer_carrier_active_ns: 10_200_000,
    product_carrier_wall_ns: 500_000,
    midpoint_active_ns: 2_600_000,
    host_work_ns: 2_500_000,
    complete_five_x_ns: FIVE_X_CAP_NS,
    complete_eight_x_ns: EIGHT_X_CAP_NS,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelError {
    Abi(SpartanShiftSuccessorAbiError),
    InvalidHighTiles,
    InvalidRate,
    ArithmeticOverflow,
}

impl From<SpartanShiftSuccessorAbiError> for ModelError {
    fn from(error: SpartanShiftSuccessorAbiError) -> Self {
        Self::Abi(error)
    }
}

pub fn target_plan(midpoint: MidpointPlan) -> Result<WorkPlan, ModelError> {
    work_plan(
        SpartanShiftSuccessorGeometry::new(TARGET_ROWS)?,
        midpoint,
        1,
    )
}

pub fn work_plan(
    geometry: SpartanShiftSuccessorGeometry,
    midpoint: MidpointPlan,
    high_tiles: usize,
) -> Result<WorkPlan, ModelError> {
    if high_tiles == 0
        || !high_tiles.is_power_of_two()
        || !geometry.suffix_elements.is_multiple_of(high_tiles)
    {
        return Err(ModelError::InvalidHighTiles);
    }
    geometry.outer_partial_params(geometry.suffix_elements / high_tiles)?;
    let rows = geometry.rows as u128;
    let prefix = geometry.prefix_elements as u128;
    let suffix = geometry.suffix_elements as u128;

    let successor_rows = rows
        .checked_sub(prefix)
        .ok_or(ModelError::ArithmeticOverflow)?;
    let outer_half_terms = checked_mul(2, checked_add(rows, successor_rows)?)?;
    let outer_flag_adds = outer_half_terms;
    let product_flag_adds = checked_add(rows, successor_rows)?;

    let outer_output_bytes =
        tiled_output_bytes(prefix, OUTER_COMPONENT_TABLES as u128, high_tiles as u128)?;
    let product_output_bytes =
        tiled_output_bytes(prefix, PRODUCT_COMPONENT_TABLES as u128, high_tiles as u128)?;
    let high_weight_bytes = checked_mul(suffix, u128::from(FIELD_BYTES))?;
    let outer_native_bytes = checked_mul(rows, 16)?;
    let two_flag_plane_bytes = checked_mul(geometry.flag_words as u128, 2 * size_of_u32() as u128)?;
    let one_flag_plane_bytes = checked_mul(geometry.flag_words as u128, size_of_u32() as u128)?;

    let outer_carrier = PhaseWork {
        name: "outer component carrier",
        half_width_terms: outer_half_terms,
        full_products: 0,
        selected_field_adds_max: outer_flag_adds,
        logical_bytes: checked_sum(&[
            outer_native_bytes,
            two_flag_plane_bytes,
            high_weight_bytes,
            outer_output_bytes,
        ])?,
    };
    let product_carrier = PhaseWork {
        name: "product component carrier",
        half_width_terms: 0,
        full_products: 0,
        selected_field_adds_max: product_flag_adds,
        logical_bytes: checked_sum(&[
            one_flag_plane_bytes,
            high_weight_bytes,
            product_output_bytes,
        ])?,
    };

    let carrier_fields = checked_mul(
        prefix,
        (OUTER_COMPONENT_TABLES + PRODUCT_COMPONENT_TABLES) as u128,
    )?;
    let q_fields = checked_mul(prefix, PREFIX_Q_TABLES as u128)?;
    let q_combine = PhaseWork {
        name: "ten-to-four Q combine",
        half_width_terms: 0,
        full_products: checked_mul(prefix, 8)?,
        selected_field_adds_max: checked_mul(prefix, 6)?,
        logical_bytes: checked_mul(
            checked_add(carrier_fields, q_fields)?,
            u128::from(FIELD_BYTES),
        )?,
    };

    let prefix_products = checked_mul(prefix, 16)?
        .checked_sub(24)
        .ok_or(ModelError::ArithmeticOverflow)?;
    let prefix_host = PhaseWork {
        name: "prefix host ladder",
        half_width_terms: 0,
        full_products: prefix_products,
        selected_field_adds_max: 0,
        logical_bytes: 0,
    };

    let midpoint_numeric_columns = match midpoint {
        MidpointPlan::BorrowInstructionInputUpc => 1,
        MidpointPlan::SelfContained => 2,
    };
    let midpoint_output_columns = match midpoint {
        MidpointPlan::BorrowInstructionInputUpc => MIDPOINT_RESIDUAL_TABLES,
        MidpointPlan::SelfContained => MIDPOINT_FULL_TABLES,
    };
    let midpoint_native_bytes = checked_mul(rows, midpoint_numeric_columns * 8)?;
    let transient_upc_projection_bytes = checked_mul(rows, 8)?;
    let midpoint_flag_bytes = checked_mul(
        geometry.flag_words as u128,
        FLAG_PLANES as u128 * size_of_u32() as u128,
    )?;
    let midpoint_weight_bytes = checked_mul(prefix, u128::from(FIELD_BYTES))?;
    let midpoint_output_bytes = checked_mul(
        checked_mul(suffix, midpoint_output_columns as u128)?,
        u128::from(FIELD_BYTES),
    )?;
    let midpoint_fold = PhaseWork {
        name: "midpoint native fold",
        half_width_terms: checked_mul(rows, midpoint_numeric_columns)?,
        full_products: 0,
        selected_field_adds_max: checked_mul(rows, FLAG_PLANES as u128)?,
        logical_bytes: checked_sum(&[
            midpoint_native_bytes,
            midpoint_flag_bytes,
            midpoint_weight_bytes,
            midpoint_output_bytes,
        ])?,
    };

    let suffix_products = checked_mul(suffix, 19)?
        .checked_sub(19)
        .ok_or(ModelError::ArithmeticOverflow)?;
    let suffix_host = PhaseWork {
        name: "suffix host ladder",
        half_width_terms: 0,
        full_products: suffix_products,
        selected_field_adds_max: 0,
        logical_bytes: 0,
    };

    let host_handoff_fields = checked_add(
        checked_mul(prefix, PREFIX_Q_TABLES as u128)?,
        checked_mul(suffix, MIDPOINT_FULL_TABLES as u128)?,
    )?;

    Ok(WorkPlan {
        geometry,
        midpoint,
        high_tiles,
        outer_carrier,
        product_carrier,
        q_combine,
        prefix_host,
        midpoint_fold,
        suffix_host,
        outer_logical_source_bytes: checked_add(outer_native_bytes, two_flag_plane_bytes)?,
        transient_upc_projection_bytes,
        producer_projection_bytes: checked_sum(&[
            transient_upc_projection_bytes,
            checked_mul(rows, 8)?,
            midpoint_flag_bytes,
        ])?,
        host_handoff_bytes: checked_mul(host_handoff_fields, u128::from(FIELD_BYTES))?,
    })
}

impl WorkPlan {
    pub fn phases(self) -> [PhaseWork; 6] {
        [
            self.outer_carrier,
            self.product_carrier,
            self.q_combine,
            self.prefix_host,
            self.midpoint_fold,
            self.suffix_host,
        ]
    }

    pub fn gross_half_width_terms(self) -> Result<u128, ModelError> {
        checked_add(
            self.outer_carrier.half_width_terms,
            self.midpoint_fold.half_width_terms,
        )
    }

    /// Shift-attributed terms after successor tables replace upstream openings.
    pub fn incremental_half_width_terms(self) -> Result<u128, ModelError> {
        let current_outer_numeric = checked_mul(self.geometry.rows as u128, 2)?;
        checked_add(current_outer_numeric, self.midpoint_fold.half_width_terms)
    }

    pub fn full_products(self) -> Result<u128, ModelError> {
        self.phases()
            .into_iter()
            .try_fold(0, |total, phase| checked_add(total, phase.full_products))
    }

    pub fn selected_field_adds_max(self) -> Result<u128, ModelError> {
        self.phases().into_iter().try_fold(0, |total, phase| {
            checked_add(total, phase.selected_field_adds_max)
        })
    }

    pub fn kernel_logical_device_bytes(self) -> Result<u128, ModelError> {
        self.phases()
            .into_iter()
            .try_fold(0, |total, phase| checked_add(total, phase.logical_bytes))
    }

    pub fn logical_device_bytes(self) -> Result<u128, ModelError> {
        checked_add(
            self.kernel_logical_device_bytes()?,
            self.producer_projection_bytes,
        )
    }

    pub fn fresh_scan_compulsory_device_bytes(self, split_outer: bool) -> Result<u128, ModelError> {
        let instruction_passes = if split_outer { 2 } else { 1 };
        let physical_outer_row_bytes = checked_add(
            checked_mul(u128::from(INSTRUCTION_INPUT_ROW_BYTES), instruction_passes)?,
            u128::from(SPARTAN_OUTER_RESIDUAL_ROW_BYTES),
        )?;
        let physical_outer_source =
            checked_mul(self.geometry.rows as u128, physical_outer_row_bytes)?;
        let unused_projection = match self.midpoint {
            MidpointPlan::BorrowInstructionInputUpc => self.transient_upc_projection_bytes,
            MidpointPlan::SelfContained => 0,
        };
        self.logical_device_bytes()?
            .checked_sub(self.outer_logical_source_bytes)
            .and_then(|bytes| bytes.checked_sub(unused_projection))
            .and_then(|bytes| bytes.checked_add(physical_outer_source))
            .ok_or(ModelError::ArithmeticOverflow)
    }

    pub fn projection_write_floor_ns(self) -> Result<u64, ModelError> {
        ceil_rate(
            self.producer_projection_bytes,
            RETAINED_COPY_BYTES_PER_SECOND,
        )
    }

    pub fn nonoverlapped_projection_compute_floor_ns(
        self,
        half_terms_per_second: u64,
    ) -> Result<u64, ModelError> {
        self.projection_write_floor_ns()?
            .checked_add(ceil_rate(
                self.gross_half_width_terms()?,
                half_terms_per_second,
            )?)
            .ok_or(ModelError::ArithmeticOverflow)
    }

    pub fn roof(
        self,
        boundary: AttributionBoundary,
        half_terms_per_second: u64,
    ) -> Result<RoofBounds, ModelError> {
        let terms = match boundary {
            AttributionBoundary::GrossCoMaterializedCompact
            | AttributionBoundary::GrossFreshFusedOuter
            | AttributionBoundary::GrossFreshSplitOuter => self.gross_half_width_terms()?,
            AttributionBoundary::ResidentPiopIncremental => self.incremental_half_width_terms()?,
        };
        let bytes = match boundary {
            AttributionBoundary::GrossCoMaterializedCompact => self.logical_device_bytes()?,
            AttributionBoundary::GrossFreshFusedOuter => {
                self.fresh_scan_compulsory_device_bytes(false)?
            }
            AttributionBoundary::GrossFreshSplitOuter => {
                self.fresh_scan_compulsory_device_bytes(true)?
            }
            AttributionBoundary::ResidentPiopIncremental => {
                // The actual delta depends on which upstream opening loads disappear.
                0
            }
        };
        roof_bounds(terms, bytes, half_terms_per_second)
    }
}

pub fn roof_bounds(
    half_width_terms: u128,
    compulsory_bytes: u128,
    half_terms_per_second: u64,
) -> Result<RoofBounds, ModelError> {
    if half_terms_per_second == 0 {
        return Err(ModelError::InvalidRate);
    }
    let compute_floor_ns = ceil_rate(half_width_terms, half_terms_per_second)?;
    let traffic_floor_ns = ceil_rate(compulsory_bytes, RETAINED_COPY_BYTES_PER_SECOND)?;
    let binding_floor_ns = compute_floor_ns.max(traffic_floor_ns);
    let eighty_percent_bar_ns = ceil_div(
        u128::from(binding_floor_ns)
            .checked_mul(PERMILLE)
            .ok_or(ModelError::ArithmeticOverflow)?,
        u128::from(ROOF_EFFICIENCY_PERMILLE),
    )?;
    Ok(RoofBounds {
        compute_floor_ns,
        traffic_floor_ns,
        binding_floor_ns,
        eighty_percent_bar_ns,
    })
}

fn tiled_output_bytes(
    prefix_elements: u128,
    columns: u128,
    high_tiles: u128,
) -> Result<u128, ModelError> {
    let final_bytes = checked_mul(
        checked_mul(prefix_elements, columns)?,
        u128::from(FIELD_BYTES),
    )?;
    if high_tiles == 1 {
        return Ok(final_bytes);
    }
    let partial_bytes = checked_mul(final_bytes, high_tiles)?;
    checked_add(checked_mul(partial_bytes, 2)?, final_bytes)
}

fn ceil_rate(work: u128, per_second: u64) -> Result<u64, ModelError> {
    if work == 0 {
        return Ok(0);
    }
    if per_second == 0 {
        return Err(ModelError::InvalidRate);
    }
    let numerator = work
        .checked_mul(NANOS_PER_SECOND)
        .ok_or(ModelError::ArithmeticOverflow)?;
    ceil_div(numerator, u128::from(per_second))
}

fn ceil_div(numerator: u128, denominator: u128) -> Result<u64, ModelError> {
    let rounded = numerator
        .checked_add(denominator - 1)
        .ok_or(ModelError::ArithmeticOverflow)?
        / denominator;
    u64::try_from(rounded).map_err(|_| ModelError::ArithmeticOverflow)
}

fn checked_add(lhs: u128, rhs: u128) -> Result<u128, ModelError> {
    lhs.checked_add(rhs).ok_or(ModelError::ArithmeticOverflow)
}

fn checked_mul(lhs: u128, rhs: u128) -> Result<u128, ModelError> {
    lhs.checked_mul(rhs).ok_or(ModelError::ArithmeticOverflow)
}

fn checked_sum(values: &[u128]) -> Result<u128, ModelError> {
    values
        .iter()
        .try_fold(0, |total, &value| checked_add(total, value))
}

const fn size_of_u32() -> usize {
    core::mem::size_of::<u32>()
}
