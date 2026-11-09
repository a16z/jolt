//! Optional base-field arithmetic after the canonical packing plan is checked.

use akita_error::AkitaError;
use akita_pcs::custom_source::SubringCoefficientPackingPlan;
use akita_types::FpExtEncoding;
use jolt_field::ExtField;

use super::source::TraceOneHotColumn;
use crate::AkitaField;

/// Checked base-field opening weights in the canonical PCS coordinate order.
/// Extension-field plans retain the CPU path. Output is column / block /
/// subring coefficient; the sole extension coordinate is zero.
#[derive(Clone, Copy, Debug)]
pub struct TraceOneHotPackingPlan<'a> {
    ring_dimension: usize,
    segment_rings: usize,
    num_positions: usize,
    blocks_per_column: usize,
    subring_dimension: usize,
    position_weights: &'a [AkitaField],
    packing_weights: &'a [AkitaField],
}

impl<'a> TraceOneHotPackingPlan<'a> {
    pub fn ring_dimension(self) -> usize {
        self.ring_dimension
    }
    pub fn segment_rings(self) -> usize {
        self.segment_rings
    }
    pub fn num_positions(self) -> usize {
        self.num_positions
    }
    pub fn blocks_per_column(self) -> usize {
        self.blocks_per_column
    }
    pub fn subring_dimension(self) -> usize {
        self.subring_dimension
    }
    pub fn position_weights(self) -> &'a [AkitaField] {
        self.position_weights
    }
    pub fn packing_weights(self) -> &'a [AkitaField] {
        self.packing_weights
    }
}

/// Called only after plan/domain/extension/weight-width validation by traversal.
pub(super) fn try_compute<E, const D: usize>(
    source: &TraceOneHotColumn,
    plan: SubringCoefficientPackingPlan<'_, E>,
    segment_rings: usize,
) -> Result<Option<Vec<Vec<AkitaField>>>, AkitaError>
where
    E: ExtField<AkitaField> + FpExtEncoding<AkitaField>,
{
    let Some(kernel) = &source.compute_kernel else {
        return Ok(None);
    };
    if E::DEGREE != 1 {
        return Ok(None);
    }
    // Use the canonical encoding contract, without layout casts or assuming
    // that every degree-one extension exposes a base scalar.
    let convert = |values: &[E]| {
        values
            .iter()
            .map(FpExtEncoding::degree_one_base)
            .collect::<Option<Vec<AkitaField>>>()
    };
    let (Some(positions), Some(packing)) = (
        convert(plan.tables.position_weights()),
        convert(plan.tables.packing_weights()),
    ) else {
        return Ok(None);
    };
    let point = plan.tables.point();
    let geometry = point.geometry();
    let output = kernel.coefficient_packing(
        source.rows.as_ref(),
        source.one_hot_k,
        TraceOneHotPackingPlan {
            ring_dimension: D,
            segment_rings,
            num_positions: point.num_positions_per_block(),
            blocks_per_column: point.num_live_blocks(),
            subring_dimension: geometry.challenge_subring_dimension(),
            position_weights: &positions,
            packing_weights: &packing,
        },
    )?;
    if let Some(columns) = &output {
        let expected = point
            .num_live_blocks()
            .checked_mul(geometry.partial_base_field_width())
            .ok_or_else(|| AkitaError::InvalidInput("packing output width overflows".into()))?;
        if columns.len() != source.rows.num_columns()
            || columns.iter().any(|column| column.len() != expected)
        {
            return Err(AkitaError::InvalidInput(
                "compute packing output disagrees with canonical geometry".into(),
            ));
        }
    }
    Ok(output)
}
