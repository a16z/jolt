use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_prover::compute::{OpeningFoldOutput, OpeningFoldPlan};
use rayon::prelude::*;

use super::source::TracePackedOneHot;
use super::traversal::{
    row_is_committed, trace_block_part_range, trace_block_task_parts, validate_block_geometry,
    visit_segment_ring_range, visit_segment_ring_row_range,
};
use crate::AkitaField;

enum PackedOpeningWeights<'a, const D: usize> {
    Base {
        live_block_weights: &'a [AkitaField],
        position_weights: &'a [AkitaField],
    },
    Subfield {
        live_block_weights: Vec<CyclotomicRing<AkitaField, D>>,
        position_weights: Vec<CyclotomicRing<AkitaField, D>>,
    },
}

pub(super) fn opening_fold_packed<const D: usize>(
    source: &TracePackedOneHot,
    plan: OpeningFoldPlan<'_, AkitaField>,
) -> Result<OpeningFoldOutput<AkitaField, D>, AkitaError> {
    let (num_positions, weights) = match plan {
        OpeningFoldPlan::Base {
            live_block_weights,
            position_weights,
            num_positions_per_block,
        } => (
            num_positions_per_block,
            PackedOpeningWeights::Base {
                live_block_weights,
                position_weights,
            },
        ),
        OpeningFoldPlan::Subfield {
            multipliers,
            num_positions_per_block,
        } => (
            num_positions_per_block,
            PackedOpeningWeights::Subfield {
                live_block_weights: multipliers.materialize_fold_rings::<D>()?,
                position_weights: multipliers.materialize_position_rings::<D>()?,
            },
        ),
    };
    let weight_kind = match &weights {
        PackedOpeningWeights::Base { .. } => "base",
        PackedOpeningWeights::Subfield { .. } => "subfield",
    };
    let _span = tracing::info_span!(
        "TracePackedOneHot::evaluate_and_fold",
        ring_dimension = D,
        one_hot_k = source.one_hot_k,
        rows = source.rows.num_rows(),
        columns = source.rows.num_columns(),
        column_capacity = source.column_capacity,
        positions_per_block = num_positions,
        weight_kind,
    )
    .entered();
    let segment_rings = source.segment_ring_elems::<D>()?;
    let (_, num_blocks) =
        validate_block_geometry(segment_rings, source.column_capacity, num_positions)?;
    let (live_weights, position_weights) = match &weights {
        PackedOpeningWeights::Base {
            live_block_weights,
            position_weights,
        } => (live_block_weights.len(), position_weights.len()),
        PackedOpeningWeights::Subfield {
            live_block_weights,
            position_weights,
        } => (live_block_weights.len(), position_weights.len()),
    };
    if live_weights != num_blocks || position_weights != num_positions {
        return Err(AkitaError::InvalidInput(format!(
            "trace one-hot opening weights ({live_weights}, {position_weights}) do not match block geometry ({num_blocks}, {num_positions})"
        )));
    }
    let folded = if segment_rings >= num_positions {
        let blocks_per_column = segment_rings / num_positions;
        let parts = trace_block_task_parts::<D>(source.one_hot_k, num_positions, blocks_per_column);
        let ring_alignment = (source.one_hot_k / D).max(1);
        let num_columns = source.rows.num_columns();
        let _accumulate_span = tracing::info_span!(
            "trace_onehot_evaluate_fold_accumulate",
            num_blocks,
            blocks_per_column,
            task_parts = parts,
            tasks = blocks_per_column * parts,
            active_columns = num_columns,
            rows_per_ring = (D / source.one_hot_k).max(1),
            weight_kind,
        )
        .entered();
        let partials = (0..blocks_per_column * parts)
            .into_par_iter()
            .map(|task| {
                let trace_block = task / parts;
                let part = task % parts;
                let block_ring_start = trace_block * num_positions;
                let (part_start, part_end) =
                    trace_block_part_range(num_positions, ring_alignment, part, parts);
                let ring_start = block_ring_start + part_start;
                let ring_end = block_ring_start + part_end;
                let mut folded = vec![CyclotomicRing::zero(); num_columns];
                if source.one_hot_k < D {
                    visit_segment_ring_row_range::<D>(
                        source,
                        ring_start,
                        ring_end,
                        |ring, selected_rows, committed_zero_masks| {
                            let position = ring - block_ring_start;
                            match &weights {
                                PackedOpeningWeights::Base {
                                    position_weights, ..
                                } => {
                                    let weight = position_weights[position];
                                    for (row_offset, (row_indices, &committed_zero_mask)) in
                                        selected_rows
                                            .chunks_exact(num_columns)
                                            .zip(committed_zero_masks)
                                            .enumerate()
                                    {
                                        let coefficient_base = row_offset * source.one_hot_k;
                                        for (column, &hot) in row_indices.iter().enumerate() {
                                            if row_is_committed(hot, committed_zero_mask, column) {
                                                folded[column].coeffs
                                                    [coefficient_base + usize::from(hot)] += weight;
                                            }
                                        }
                                    }
                                }
                                PackedOpeningWeights::Subfield {
                                    position_weights, ..
                                } => {
                                    let weight = position_weights[position];
                                    for (row_offset, (row_indices, &committed_zero_mask)) in
                                        selected_rows
                                            .chunks_exact(num_columns)
                                            .zip(committed_zero_masks)
                                            .enumerate()
                                    {
                                        let coefficient_base = row_offset * source.one_hot_k;
                                        for (column, &hot) in row_indices.iter().enumerate() {
                                            if row_is_committed(hot, committed_zero_mask, column) {
                                                weight.shift_accumulate_into(
                                                    &mut folded[column],
                                                    coefficient_base + usize::from(hot),
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        },
                    )?;
                } else {
                    visit_segment_ring_range::<D>(
                        source,
                        ring_start,
                        ring_end,
                        |ring, contributions| {
                            let position = ring - block_ring_start;
                            match &weights {
                                PackedOpeningWeights::Base {
                                    position_weights, ..
                                } => {
                                    let weight = position_weights[position];
                                    for &(column, coefficient) in contributions {
                                        folded[column].coeffs[coefficient] += weight;
                                    }
                                }
                                PackedOpeningWeights::Subfield {
                                    position_weights, ..
                                } => {
                                    let weight = position_weights[position];
                                    for &(column, coefficient) in contributions {
                                        weight.shift_accumulate_into(
                                            &mut folded[column],
                                            coefficient,
                                        );
                                    }
                                }
                            }
                        },
                    )?;
                }
                Ok::<_, AkitaError>(folded)
            })
            .collect::<Result<Vec<_>, _>>()?;
        drop(_accumulate_span);
        let _merge_span = tracing::info_span!(
            "trace_onehot_evaluate_fold_merge_partials",
            num_blocks,
            blocks_per_column,
            task_parts = parts,
            active_columns = num_columns,
        )
        .entered();
        let mut folded = vec![CyclotomicRing::zero(); num_blocks];
        for (task, trace_folded) in partials.into_iter().enumerate() {
            let trace_block = task / parts;
            let part = task % parts;
            for column in 0..num_columns {
                let dst = &mut folded[column * blocks_per_column + trace_block];
                if part == 0 {
                    *dst = trace_folded[column];
                } else {
                    *dst += trace_folded[column];
                }
            }
        }
        folded
    } else {
        let _accumulate_span = tracing::info_span!(
            "trace_onehot_evaluate_fold_accumulate_flat",
            num_blocks,
            segment_rings,
            weight_kind,
        )
        .entered();
        let mut folded = vec![CyclotomicRing::zero(); num_blocks];
        visit_segment_ring_range::<D>(source, 0, segment_rings, |ring, contributions| {
            for &(column, coefficient) in contributions {
                let global_ring = column * segment_rings + ring;
                let block = global_ring / num_positions;
                let position = global_ring % num_positions;
                match &weights {
                    PackedOpeningWeights::Base {
                        position_weights, ..
                    } => {
                        folded[block].coeffs[coefficient] += position_weights[position];
                    }
                    PackedOpeningWeights::Subfield {
                        position_weights, ..
                    } => {
                        position_weights[position]
                            .shift_accumulate_into(&mut folded[block], coefficient);
                    }
                }
            }
        })?;
        folded
    };
    let _reduce_span = tracing::info_span!(
        "trace_onehot_evaluate_fold_reduce_blocks",
        num_blocks,
        weight_kind,
    )
    .entered();
    let eval = match &weights {
        PackedOpeningWeights::Base {
            live_block_weights, ..
        } => folded
            .iter()
            .zip(live_block_weights.iter().copied())
            .fold(CyclotomicRing::zero(), |acc, (value, weight)| {
                acc + value.scale(&weight)
            }),
        PackedOpeningWeights::Subfield {
            live_block_weights, ..
        } => folded
            .iter()
            .zip(live_block_weights)
            .fold(CyclotomicRing::zero(), |acc, (value, weight)| {
                acc + *value * *weight
            }),
    };
    Ok(OpeningFoldOutput { eval, folded })
}
