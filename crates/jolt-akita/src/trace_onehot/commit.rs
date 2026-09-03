use akita_algebra::{ring::WideCyclotomicRing, CyclotomicRing};
use akita_error::AkitaError;
use akita_prover::compute::CommitInnerPlan;
use akita_prover::{CommitInnerWitness, ComputeBackendSetup, CpuBackend};
use rayon::prelude::*;

use super::source::TracePackedOneHot;
use super::traversal::{
    flush_deferred_rank, flush_wide, row_is_committed, trace_block_part_range,
    trace_block_task_parts, try_shift_accumulate_full_rows, validate_block_geometry,
    visit_segment_ring_range, visit_segment_ring_row_range, DeferredFp128Ring,
    K16FourRowShiftGroups,
};
use super::{K256_ROW_BATCH, MAX_WIDE_ACCUMULATIONS, NO_SELECTED_ROW};
use crate::AkitaField;

pub(super) fn commit_packed<const D: usize>(
    backend: &CpuBackend,
    prepared: &<CpuBackend as ComputeBackendSetup<AkitaField>>::PreparedSetup,
    source: &TracePackedOneHot,
    plan: CommitInnerPlan,
) -> Result<CommitInnerWitness<AkitaField>, AkitaError> {
    let _span = tracing::info_span!(
        "TracePackedOneHot::commit_inner",
        ring_dimension = D,
        one_hot_k = source.one_hot_k,
        rows = source.rows.num_rows(),
        columns = source.rows.num_columns(),
        column_capacity = source.column_capacity,
        n_a = plan.n_a,
        positions_per_block = plan.num_positions_per_block,
        inner_digits = plan.num_digits_inner,
    )
    .entered();
    let _prepare_span = tracing::info_span!("trace_onehot_commit_prepare").entered();
    let segment_rings = source.segment_ring_elems::<D>()?;
    let (_, num_blocks) = validate_block_geometry(
        segment_rings,
        source.column_capacity,
        plan.num_positions_per_block,
    )?;
    let active_cols = plan
        .num_positions_per_block
        .checked_mul(plan.num_digits_inner)
        .ok_or_else(|| AkitaError::InvalidSetup("active A width overflow".to_string()))?;
    let expanded = backend.prepared_expanded_setup(prepared);
    let a_view = expanded
        .shared_matrix()
        .ring_view::<D>(plan.n_a, active_cols)?;
    let a_rows = a_view.rows().collect::<Vec<_>>();
    let max_per_ring = (D / source.one_hot_k).max(1);
    drop(_prepare_span);

    let rows = if segment_rings >= plan.num_positions_per_block {
        let blocks_per_column = segment_rings / plan.num_positions_per_block;
        debug_assert_eq!(
            blocks_per_column * plan.num_positions_per_block,
            segment_rings
        );
        let parts = trace_block_task_parts::<D>(
            source.one_hot_k,
            plan.num_positions_per_block,
            blocks_per_column,
        );
        let ring_alignment = (source.one_hot_k / D).max(1);
        let num_columns = source.rows.num_columns();
        let _accumulate_span = tracing::info_span!(
            "trace_onehot_commit_accumulate",
            num_blocks,
            blocks_per_column,
            task_parts = parts,
            tasks = blocks_per_column * parts,
            active_columns = num_columns,
            rows_per_ring = D / source.one_hot_k,
            shared_shift_groups = source.one_hot_k == 16 && matches!(D, 64 | 128 | 256),
            generic_fused_row_shifts = matches!(D / source.one_hot_k, 2 | 4 | 8 | 16 | 32),
        )
        .entered();
        let partials = (0..blocks_per_column * parts)
            .into_par_iter()
            .map(|task| {
                let trace_block = task / parts;
                let part = task % parts;
                let mut reduced = vec![CyclotomicRing::zero(); num_columns * plan.n_a];
                let block_ring_start = trace_block * plan.num_positions_per_block;
                let (part_start, part_end) = trace_block_part_range(
                    plan.num_positions_per_block,
                    ring_alignment,
                    part,
                    parts,
                );
                let ring_start = block_ring_start + part_start;
                let ring_end = block_ring_start + part_end;
                let rank_tiled_k256 = matches!(D, 64 | 128 | 256 | 512)
                    && source.one_hot_k == 256
                    && num_columns <= u32::BITS as usize;
                let mut wide = if rank_tiled_k256 {
                    Vec::new()
                } else {
                    vec![WideCyclotomicRing::zero(); num_columns * plan.n_a]
                };
                let mut budget = 0usize;
                if source.one_hot_k < D && rank_tiled_k256 {
                    let rows_per_ring = D / source.one_hot_k;
                    debug_assert_eq!(rows_per_ring, 2);
                    let row_start = ring_start * rows_per_ring;
                    let row_end = ring_end * rows_per_ring;
                    let mut hot_values = vec![0u8; K256_ROW_BATCH * num_columns];
                    let mut committed_zero_masks = vec![0u64; K256_ROW_BATCH];
                    let mut rank_deferred = vec![DeferredFp128Ring::zero(); num_columns];
                    for tile_start in (row_start..row_end).step_by(K256_ROW_BATCH) {
                        let tile_len = (row_end - tile_start).min(K256_ROW_BATCH);
                        let selected_rows = &mut hot_values[..tile_len * num_columns];
                        let zero_masks = &mut committed_zero_masks[..tile_len];
                        source.rows.fill_rows(tile_start, selected_rows);
                        source
                            .rows
                            .fill_committed_digit_zero_masks(tile_start, zero_masks);
                        for (a, a_row) in a_rows.iter().enumerate() {
                            for (row_offset, (row_indices, &committed_zero_mask)) in selected_rows
                                .chunks_exact(num_columns)
                                .zip(zero_masks.iter())
                                .enumerate()
                            {
                                let trace_row = tile_start + row_offset;
                                let ring = trace_row / rows_per_ring;
                                let position = ring - block_ring_start;
                                let a_col = position * plan.num_digits_inner;
                                let coefficient_base =
                                    (trace_row % rows_per_ring) * source.one_hot_k;
                                for (column, &hot) in row_indices.iter().enumerate() {
                                    if !row_is_committed(hot, committed_zero_mask, column) {
                                        continue;
                                    }
                                    if usize::from(hot) >= source.one_hot_k {
                                        return Err(AkitaError::InvalidInput(format!(
                                            "trace one-hot row {hot} is outside K={}",
                                            source.one_hot_k
                                        )));
                                    }
                                    rank_deferred[column].shift_accumulate(
                                        &a_row[a_col],
                                        coefficient_base + usize::from(hot),
                                    );
                                }
                            }
                            flush_deferred_rank(&mut rank_deferred, &mut reduced, plan.n_a, a);
                        }
                    }
                } else if source.one_hot_k < D {
                    let rows_per_ring = D / source.one_hot_k;
                    let mut shift_groups = (source.one_hot_k == 16
                        && rows_per_ring.is_multiple_of(4))
                    .then(|| {
                        (0..rows_per_ring / 4)
                            .map(|chunk| K16FourRowShiftGroups::new(num_columns, 4 * chunk))
                            .collect::<Option<Vec<_>>>()
                    })
                    .flatten();
                    visit_segment_ring_row_range::<D>(
                        source,
                        ring_start,
                        ring_end,
                        |ring, selected_rows, committed_zero_masks| {
                            let position = ring - block_ring_start;
                            let a_col = position * plan.num_digits_inner;
                            let grouped = shift_groups.as_mut().is_some_and(|groups| {
                                groups
                                    .iter_mut()
                                    .zip(selected_rows.chunks_exact(4 * num_columns))
                                    .zip(committed_zero_masks.chunks_exact(4))
                                    .all(|((groups, selected_rows), masks)| {
                                        groups.build(selected_rows, masks, num_columns)
                                    })
                            });
                            for (a, a_row) in a_rows.iter().enumerate() {
                                let a_wide = WideCyclotomicRing::from_ring(&a_row[a_col]);
                                if grouped {
                                    if let Some(groups) = &shift_groups {
                                        for ((groups, selected_rows), masks) in groups
                                            .iter()
                                            .zip(selected_rows.chunks_exact(4 * num_columns))
                                            .zip(committed_zero_masks.chunks_exact(4))
                                        {
                                            groups.accumulate(
                                                &a_wide,
                                                &mut wide,
                                                a,
                                                plan.n_a,
                                                selected_rows,
                                                masks,
                                                num_columns,
                                            );
                                        }
                                        continue;
                                    }
                                }
                                for column in 0..num_columns {
                                    let dst = &mut wide[column * plan.n_a + a];
                                    if try_shift_accumulate_full_rows(
                                        &a_wide,
                                        dst,
                                        selected_rows,
                                        committed_zero_masks,
                                        num_columns,
                                        column,
                                        source.one_hot_k,
                                        rows_per_ring,
                                    ) {
                                        continue;
                                    }
                                    for (row_offset, (row_indices, &committed_zero_mask)) in
                                        selected_rows
                                            .chunks_exact(num_columns)
                                            .zip(committed_zero_masks)
                                            .enumerate()
                                    {
                                        let hot = row_indices[column];
                                        if row_is_committed(hot, committed_zero_mask, column) {
                                            a_wide.shift_accumulate_into(
                                                dst,
                                                row_offset * source.one_hot_k + usize::from(hot),
                                            );
                                        }
                                    }
                                }
                            }
                            budget += rows_per_ring;
                            if budget >= MAX_WIDE_ACCUMULATIONS {
                                flush_wide(&mut wide, &mut reduced);
                                budget = 0;
                            }
                        },
                    )?;
                } else if rank_tiled_k256 {
                    // Stream one A rank at a time so its destination accumulators fit in cache.
                    let rings_per_row = source.one_hot_k / D;
                    debug_assert!(matches!(rings_per_row, 1 | 2 | 4));
                    debug_assert_eq!(ring_start % rings_per_row, 0);
                    debug_assert_eq!(ring_end % rings_per_row, 0);
                    let row_start = ring_start / rings_per_row;
                    let row_end = ring_end / rings_per_row;
                    let mut selected_rows = vec![NO_SELECTED_ROW; num_columns];
                    let mut hot_values = vec![0u8; K256_ROW_BATCH * num_columns];
                    let mut ring_masks = vec![[0u32; 4]; K256_ROW_BATCH];
                    let mut rank_deferred = vec![DeferredFp128Ring::zero(); num_columns];
                    for tile_start in (row_start..row_end).step_by(K256_ROW_BATCH) {
                        let tile_len = (row_end - tile_start).min(K256_ROW_BATCH);
                        for row_offset in 0..tile_len {
                            let row = tile_start + row_offset;
                            source.rows.fill_row(row, &mut selected_rows);
                            let committed_zero_mask = source.rows.committed_digit_zero_mask(row);
                            let masks = &mut ring_masks[row_offset];
                            *masks = [0; 4];
                            for (column, &hot) in selected_rows.iter().enumerate() {
                                if !row_is_committed(hot, committed_zero_mask, column) {
                                    continue;
                                }
                                if usize::from(hot) >= source.one_hot_k {
                                    return Err(AkitaError::InvalidInput(format!(
                                        "trace one-hot row {hot} is outside K={}",
                                        source.one_hot_k
                                    )));
                                }
                                hot_values[row_offset * num_columns + column] = hot;
                                masks[usize::from(hot) / D] |= 1 << column;
                            }
                        }
                        for (a, a_row) in a_rows.iter().enumerate() {
                            for row_offset in 0..tile_len {
                                let trace_row = tile_start + row_offset;
                                for (ring_offset, &mask) in
                                    ring_masks[row_offset][..rings_per_row].iter().enumerate()
                                {
                                    if mask == 0 {
                                        continue;
                                    }
                                    let ring = trace_row * rings_per_row + ring_offset;
                                    let position = ring - block_ring_start;
                                    let a_col = position * plan.num_digits_inner;
                                    let mut remaining = mask;
                                    while remaining != 0 {
                                        let column = remaining.trailing_zeros() as usize;
                                        remaining &= remaining - 1;
                                        let hot =
                                            hot_values[row_offset * num_columns + column] as usize;
                                        rank_deferred[column]
                                            .shift_accumulate(&a_row[a_col], hot % D);
                                    }
                                }
                            }
                            flush_deferred_rank(&mut rank_deferred, &mut reduced, plan.n_a, a);
                        }
                    }
                } else {
                    visit_segment_ring_range::<D>(
                        source,
                        ring_start,
                        ring_end,
                        |ring, contributions| {
                            if contributions.is_empty() {
                                return;
                            }
                            let position = ring - block_ring_start;
                            let a_col = position * plan.num_digits_inner;
                            for (a, a_row) in a_rows.iter().enumerate() {
                                let a_wide = WideCyclotomicRing::from_ring(&a_row[a_col]);
                                for &(column, coefficient) in contributions {
                                    a_wide.shift_accumulate_into(
                                        &mut wide[column * plan.n_a + a],
                                        coefficient,
                                    );
                                }
                            }
                            budget += max_per_ring;
                            if budget >= MAX_WIDE_ACCUMULATIONS {
                                flush_wide(&mut wide, &mut reduced);
                                budget = 0;
                            }
                        },
                    )?;
                }
                if budget != 0 {
                    flush_wide(&mut wide, &mut reduced);
                }
                Ok::<_, AkitaError>(reduced)
            })
            .collect::<Result<Vec<_>, _>>()?;
        drop(_accumulate_span);
        let _merge_span = tracing::info_span!(
            "trace_onehot_commit_merge_partials",
            num_blocks,
            blocks_per_column,
            task_parts = parts,
            active_columns = num_columns,
            n_a = plan.n_a,
        )
        .entered();
        let mut rows = vec![vec![CyclotomicRing::zero(); plan.n_a]; num_blocks];
        for (task, block_rows) in partials.into_iter().enumerate() {
            let trace_block = task / parts;
            let part = task % parts;
            for column in 0..num_columns {
                let dst = &mut rows[column * blocks_per_column + trace_block];
                let src = &block_rows[column * plan.n_a..(column + 1) * plan.n_a];
                if part == 0 {
                    dst.copy_from_slice(src);
                } else {
                    for (dst, src) in dst.iter_mut().zip(src) {
                        *dst += *src;
                    }
                }
            }
        }
        rows
    } else {
        let _accumulate_span = tracing::info_span!(
            "trace_onehot_commit_accumulate_flat",
            num_blocks,
            segment_rings,
            n_a = plan.n_a,
        )
        .entered();
        let mut wide = vec![WideCyclotomicRing::zero(); num_blocks * plan.n_a];
        let mut reduced = vec![CyclotomicRing::zero(); num_blocks * plan.n_a];
        let mut budget = 0usize;
        visit_segment_ring_range::<D>(source, 0, segment_rings, |ring, contributions| {
            for &(column, coefficient) in contributions {
                let global_ring = column * segment_rings + ring;
                let block = global_ring / plan.num_positions_per_block;
                let position = global_ring % plan.num_positions_per_block;
                let a_col = position * plan.num_digits_inner;
                for (a, a_row) in a_rows.iter().enumerate() {
                    let a_wide = WideCyclotomicRing::from_ring(&a_row[a_col]);
                    a_wide.shift_accumulate_into(&mut wide[block * plan.n_a + a], coefficient);
                }
            }
            budget += contributions.len();
            if budget >= MAX_WIDE_ACCUMULATIONS {
                flush_wide(&mut wide, &mut reduced);
                budget = 0;
            }
        })?;
        if budget != 0 {
            flush_wide(&mut wide, &mut reduced);
        }
        reduced
            .chunks_exact(plan.n_a)
            .map(<[CyclotomicRing<AkitaField, D>]>::to_vec)
            .collect()
    };

    Ok(CommitInnerWitness::from_rows(rows))
}
