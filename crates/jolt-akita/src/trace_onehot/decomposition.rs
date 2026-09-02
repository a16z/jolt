use akita_challenges::SparseChallenge;
use akita_error::AkitaError;
use akita_prover::backend::poly_helpers::{build_decompose_fold_witness, fill_rotated_challenge};
use akita_prover::DecomposeFoldWitness;
use jolt_field::One;
use rayon::prelude::*;

use super::source::TracePackedOneHot;
use super::traversal::{
    row_is_committed, validate_block_geometry, visit_segment_ring_range,
    visit_segment_ring_row_range,
};
use super::{
    DECOMPOSE_POSITION_WORKING_SET_TARGET, ROTATED_CHALLENGE_TABLE_BUDGET, TASKS_PER_RAYON_WORKER,
};
use crate::AkitaField;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DecomposeRotationMode {
    Auto,
    Compact,
    Dense,
    Sparse,
}

impl DecomposeRotationMode {
    fn from_env() -> Result<Self, AkitaError> {
        match std::env::var("JOLT_AKITA_DECOMPOSE_MODE").as_deref() {
            Ok("compact") => Ok(Self::Compact),
            Ok("dense") => Ok(Self::Dense),
            Ok("sparse") => Ok(Self::Sparse),
            Ok("auto") | Err(std::env::VarError::NotPresent) => Ok(Self::Auto),
            Ok(value) => Err(AkitaError::InvalidInput(format!(
                "JOLT_AKITA_DECOMPOSE_MODE must be auto, compact, dense, or sparse; got {value:?}"
            ))),
            Err(error) => Err(AkitaError::InvalidInput(format!(
                "failed to read JOLT_AKITA_DECOMPOSE_MODE: {error}"
            ))),
        }
    }
}

struct PreparedSparseClass {
    coefficient: i32,
    positions: Vec<u16>,
    wrap_cuts: Vec<u16>,
}

pub(super) struct PreparedSparseChallenge {
    classes: Vec<PreparedSparseClass>,
}

impl PreparedSparseChallenge {
    fn new<const D: usize>(challenge: &SparseChallenge) -> Result<Self, AkitaError> {
        if D > usize::from(u16::MAX) + 1 {
            return Err(AkitaError::InvalidInput(format!(
                "prepared sparse rotations require D <= {}; got {D}",
                usize::from(u16::MAX) + 1
            )));
        }
        let mut grouped = Vec::<(i8, Vec<u16>)>::new();
        for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
            let position = u16::try_from(position).map_err(|_| {
                AkitaError::InvalidInput(format!(
                    "sparse challenge position {position} does not fit u16"
                ))
            })?;
            if let Some((_, positions)) = grouped
                .iter_mut()
                .find(|(existing, _)| *existing == coefficient)
            {
                positions.push(position);
            } else {
                grouped.push((coefficient, vec![position]));
            }
        }
        grouped.sort_unstable_by_key(|(coefficient, _)| *coefficient);
        let classes = grouped
            .into_iter()
            .map(|(coefficient, mut positions)| {
                positions.sort_unstable();
                let wrap_cuts = (0..D)
                    .map(|shift| {
                        positions.partition_point(|&position| usize::from(position) < D - shift)
                            as u16
                    })
                    .collect();
                PreparedSparseClass {
                    coefficient: i32::from(coefficient),
                    positions,
                    wrap_cuts,
                }
            })
            .collect();
        Ok(Self { classes })
    }
}

pub(super) enum PreparedRotations<const D: usize> {
    Compact(Vec<[i8; D]>),
    Dense(Vec<[i16; D]>),
    Sparse(Vec<PreparedSparseChallenge>),
}

impl<const D: usize> PreparedRotations<D> {
    fn is_dense(&self) -> bool {
        matches!(self, Self::Dense(_))
    }
}

fn active_challenge_index(
    prepared_block: usize,
    blocks_per_column: Option<usize>,
    num_columns: usize,
) -> usize {
    blocks_per_column.map_or(prepared_block, |blocks_per_column| {
        let trace_block = prepared_block / num_columns;
        let column = prepared_block % num_columns;
        column * blocks_per_column + trace_block
    })
}

pub(super) fn prepare_rotations<const D: usize>(
    challenges: &[SparseChallenge],
    blocks_per_column: Option<usize>,
    num_columns: usize,
    mode: DecomposeRotationMode,
) -> Result<PreparedRotations<D>, AkitaError> {
    let prepared_blocks = blocks_per_column.map_or(challenges.len(), |blocks_per_column| {
        blocks_per_column * num_columns
    });
    let dense_bytes = prepared_blocks
        .checked_mul(D)
        .and_then(|rows| rows.checked_mul(std::mem::size_of::<[i16; D]>()))
        .ok_or_else(|| {
            AkitaError::InvalidInput("dense rotation table size overflow".to_string())
        })?;
    if mode == DecomposeRotationMode::Compact || (mode == DecomposeRotationMode::Auto && D == 128) {
        let compact = (0..prepared_blocks)
            .into_par_iter()
            .map(|prepared_block| {
                let challenge = &challenges
                    [active_challenge_index(prepared_block, blocks_per_column, num_columns)];
                let mut dense = [0i8; D];
                for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
                    dense[position as usize] = coefficient;
                }
                dense
            })
            .collect();
        return Ok(PreparedRotations::Compact(compact));
    }
    let use_dense = match mode {
        DecomposeRotationMode::Auto => D == 64 && dense_bytes <= ROTATED_CHALLENGE_TABLE_BUDGET,
        DecomposeRotationMode::Compact => unreachable!("compact rotations returned above"),
        DecomposeRotationMode::Dense => {
            if dense_bytes > ROTATED_CHALLENGE_TABLE_BUDGET {
                return Err(AkitaError::InvalidInput(format!(
                    "forced dense decompose rotation table requires {dense_bytes} bytes, exceeding \
                     the {ROTATED_CHALLENGE_TABLE_BUDGET}-byte budget"
                )));
            }
            true
        }
        DecomposeRotationMode::Sparse => false,
    };
    if use_dense {
        let mut rotated = vec![[0i16; D]; prepared_blocks * D];
        rotated
            .par_chunks_mut(D)
            .enumerate()
            .for_each(|(prepared_block, table)| {
                let challenge = &challenges
                    [active_challenge_index(prepared_block, blocks_per_column, num_columns)];
                fill_rotated_challenge(table, challenge);
            });
        Ok(PreparedRotations::Dense(rotated))
    } else {
        let prepared = (0..prepared_blocks)
            .into_par_iter()
            .map(|prepared_block| {
                PreparedSparseChallenge::new::<D>(
                    &challenges
                        [active_challenge_index(prepared_block, blocks_per_column, num_columns)],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PreparedRotations::Sparse(prepared))
    }
}

#[inline(always)]
fn add_rotated_sparse<const D: usize>(
    dst: &mut [i32; D],
    challenge: &PreparedSparseChallenge,
    shift: usize,
) {
    for class in &challenge.classes {
        let cut = usize::from(class.wrap_cuts[shift]);
        let coefficient = class.coefficient;
        for &position in &class.positions[..cut] {
            dst[usize::from(position) + shift] += coefficient;
        }
        for &position in &class.positions[cut..] {
            dst[usize::from(position) + shift - D] -= coefficient;
        }
    }
}

#[inline(always)]
fn add_rotated_dense<const D: usize>(dst: &mut [i32; D], rotated: &[i16; D]) {
    for (dst, &value) in dst.iter_mut().zip(rotated) {
        *dst += i32::from(value);
    }
}

#[inline(always)]
fn add_rotated_compact<const D: usize>(dst: &mut [i32; D], dense: &[i8; D], shift: usize) {
    let split = D - shift;
    for (dst, &value) in dst[shift..].iter_mut().zip(&dense[..split]) {
        *dst += i32::from(value);
    }
    for (dst, &value) in dst[..shift].iter_mut().zip(&dense[split..]) {
        *dst -= i32::from(value);
    }
}

#[inline(always)]
fn add_rotated_dense_tables<const D: usize, const N: usize>(
    dst: &mut [i32; D],
    tables: [&[i16; D]; N],
) {
    for coefficient in 0..D {
        let mut sum = 0i32;
        for table in tables {
            sum += i32::from(table[coefficient]);
        }
        dst[coefficient] += sum;
    }
}

#[inline(always)]
fn add_rotated<const D: usize>(
    dst: &mut [i32; D],
    rotations: &PreparedRotations<D>,
    prepared_block: usize,
    coefficient: usize,
) {
    match rotations {
        PreparedRotations::Compact(challenges) => {
            add_rotated_compact(dst, &challenges[prepared_block], coefficient);
        }
        PreparedRotations::Dense(rotated) => {
            add_rotated_dense(dst, &rotated[prepared_block * D + coefficient]);
        }
        PreparedRotations::Sparse(challenges) => {
            add_rotated_sparse(dst, &challenges[prepared_block], coefficient);
        }
    }
}

#[inline(always)]
fn add_rotated_dense_rows<const D: usize>(
    dst: &mut [i32; D],
    rotated: &[[i16; D]],
    prepared_block: usize,
    coefficients: &[usize],
) {
    let table = |coefficient| &rotated[prepared_block * D + coefficient];
    let mut remaining = coefficients;
    while remaining.len() >= 8 {
        add_rotated_dense_tables(
            dst,
            [
                table(remaining[0]),
                table(remaining[1]),
                table(remaining[2]),
                table(remaining[3]),
                table(remaining[4]),
                table(remaining[5]),
                table(remaining[6]),
                table(remaining[7]),
            ],
        );
        remaining = &remaining[8..];
    }
    match remaining {
        [] => {}
        [c0] => add_rotated_dense(dst, table(*c0)),
        [c0, c1] => add_rotated_dense_tables(dst, [table(*c0), table(*c1)]),
        [c0, c1, c2] => {
            add_rotated_dense_tables(dst, [table(*c0), table(*c1), table(*c2)]);
        }
        [c0, c1, c2, c3] => {
            add_rotated_dense_tables(dst, [table(*c0), table(*c1), table(*c2), table(*c3)]);
        }
        [c0, c1, c2, c3, c4] => add_rotated_dense_tables(
            dst,
            [table(*c0), table(*c1), table(*c2), table(*c3), table(*c4)],
        ),
        [c0, c1, c2, c3, c4, c5] => add_rotated_dense_tables(
            dst,
            [
                table(*c0),
                table(*c1),
                table(*c2),
                table(*c3),
                table(*c4),
                table(*c5),
            ],
        ),
        [c0, c1, c2, c3, c4, c5, c6] => add_rotated_dense_tables(
            dst,
            [
                table(*c0),
                table(*c1),
                table(*c2),
                table(*c3),
                table(*c4),
                table(*c5),
                table(*c6),
            ],
        ),
        _ => unreachable!("eight-entry batches leave at most seven contributions"),
    }
}

#[inline(always)]
fn add_rotated_rows<const D: usize>(
    dst: &mut [i32; D],
    rotations: &PreparedRotations<D>,
    prepared_block: usize,
    coefficients: &[usize],
) {
    match rotations {
        PreparedRotations::Compact(challenges) => {
            let challenge = &challenges[prepared_block];
            for &coefficient in coefficients {
                add_rotated_compact(dst, challenge, coefficient);
            }
        }
        PreparedRotations::Dense(rotated) => {
            add_rotated_dense_rows(dst, rotated, prepared_block, coefficients);
        }
        PreparedRotations::Sparse(challenges) => {
            let challenge = &challenges[prepared_block];
            for &coefficient in coefficients {
                add_rotated_sparse(dst, challenge, coefficient);
            }
        }
    }
}

#[inline(always)]
fn add_rotated_dense_contributions<const D: usize>(
    dst: &mut [i32; D],
    rotated: &[[i16; D]],
    contributions: &[(usize, usize)],
    table_index: impl Fn(usize, usize) -> usize + Copy,
) {
    let table =
        |&(column, coefficient): &(usize, usize)| &rotated[table_index(column, coefficient)];
    let mut remaining = contributions;
    while remaining.len() >= 8 {
        add_rotated_dense_tables(
            dst,
            [
                table(&remaining[0]),
                table(&remaining[1]),
                table(&remaining[2]),
                table(&remaining[3]),
                table(&remaining[4]),
                table(&remaining[5]),
                table(&remaining[6]),
                table(&remaining[7]),
            ],
        );
        remaining = &remaining[8..];
    }
    match remaining {
        [] => {}
        [entry0] => add_rotated_dense(dst, table(entry0)),
        [entry0, entry1] => {
            add_rotated_dense_tables(dst, [table(entry0), table(entry1)]);
        }
        [entry0, entry1, entry2] => {
            add_rotated_dense_tables(dst, [table(entry0), table(entry1), table(entry2)]);
        }
        [entry0, entry1, entry2, entry3] => add_rotated_dense_tables(
            dst,
            [table(entry0), table(entry1), table(entry2), table(entry3)],
        ),
        [entry0, entry1, entry2, entry3, entry4] => add_rotated_dense_tables(
            dst,
            [
                table(entry0),
                table(entry1),
                table(entry2),
                table(entry3),
                table(entry4),
            ],
        ),
        [entry0, entry1, entry2, entry3, entry4, entry5] => add_rotated_dense_tables(
            dst,
            [
                table(entry0),
                table(entry1),
                table(entry2),
                table(entry3),
                table(entry4),
                table(entry5),
            ],
        ),
        [entry0, entry1, entry2, entry3, entry4, entry5, entry6] => {
            add_rotated_dense_tables(
                dst,
                [
                    table(entry0),
                    table(entry1),
                    table(entry2),
                    table(entry3),
                    table(entry4),
                    table(entry5),
                    table(entry6),
                ],
            );
        }
        _ => unreachable!("eight-entry batches leave at most seven contributions"),
    }
}

fn fill_compact_rotation_table<const D: usize>(table: &mut [[i16; D]], dense: &[i8; D]) {
    debug_assert_eq!(table.len(), D);
    for (shift, row) in table.iter_mut().enumerate() {
        let split = D - shift;
        for (dst, &value) in row[shift..].iter_mut().zip(&dense[..split]) {
            *dst = i16::from(value);
        }
        for (dst, &value) in row[..shift].iter_mut().zip(&dense[split..]) {
            *dst = -i16::from(value);
        }
    }
}

#[inline(always)]
fn add_rotated_contributions<const D: usize>(
    dst: &mut [i32; D],
    contributions: &[(usize, usize)],
    rotations: &PreparedRotations<D>,
    trace_block: usize,
    num_columns: usize,
) {
    match rotations {
        PreparedRotations::Compact(challenges) => {
            let mut sum = [0i32; D];
            for &(column, coefficient) in contributions {
                add_rotated_compact(
                    &mut sum,
                    &challenges[trace_block * num_columns + column],
                    coefficient,
                );
            }
            for (dst, value) in dst.iter_mut().zip(sum) {
                *dst += value;
            }
        }
        PreparedRotations::Dense(rotated) => {
            add_rotated_dense_contributions(dst, rotated, contributions, |column, coefficient| {
                ((trace_block * num_columns + column) * D) + coefficient
            });
        }
        PreparedRotations::Sparse(challenges) => {
            for &(column, coefficient) in contributions {
                add_rotated_sparse(
                    dst,
                    &challenges[trace_block * num_columns + column],
                    coefficient,
                );
            }
        }
    }
}

pub(super) fn decompose_fold_packed_with_mode<const D: usize>(
    source: &TracePackedOneHot,
    challenges: &[SparseChallenge],
    num_positions: usize,
    num_digits: usize,
    rotation_mode: DecomposeRotationMode,
) -> Result<DecomposeFoldWitness<AkitaField>, AkitaError> {
    let _span = tracing::info_span!(
        "TracePackedOneHot::decompose_fold",
        ring_dimension = D,
        rows = source.rows.num_rows(),
        columns = source.rows.num_columns(),
        column_capacity = source.column_capacity,
        num_positions,
        num_digits,
    )
    .entered();
    if num_digits == 0 {
        return Err(AkitaError::InvalidInput(
            "trace one-hot decompose fold requires at least one digit".to_string(),
        ));
    }
    let segment_rings = source.segment_ring_elems::<D>()?;
    let (_, num_blocks) =
        validate_block_geometry(segment_rings, source.column_capacity, num_positions)?;
    if challenges.len() != num_blocks {
        return Err(AkitaError::InvalidSize {
            expected: num_blocks,
            actual: challenges.len(),
        });
    }
    for challenge in challenges {
        challenge.validate::<D>()?;
    }
    let blocks_per_column = (segment_rings >= num_positions).then(|| segment_rings / num_positions);
    let rotation_blocks = blocks_per_column.map_or(challenges.len(), |blocks_per_column| {
        blocks_per_column * source.rows.num_columns()
    });
    let rotation_table_bytes = rotation_blocks
        .saturating_mul(D)
        .saturating_mul(std::mem::size_of::<[i16; D]>());
    let rotation_span = tracing::info_span!(
        "trace_onehot_decompose_prepare_rotations",
        challenge_blocks = challenges.len(),
        rotation_table_bytes,
        table_budget_bytes = ROTATED_CHALLENGE_TABLE_BUDGET,
        requested_mode = ?rotation_mode,
        dense = tracing::field::Empty,
    );
    let rotation_guard = rotation_span.enter();
    let rotations = prepare_rotations::<D>(
        challenges,
        blocks_per_column,
        source.rows.num_columns(),
        rotation_mode,
    );
    let rotations = rotations?;
    let _ = rotation_span.record("dense", rotations.is_dense());
    drop(rotation_guard);
    let compressed = if segment_rings >= num_positions {
        let blocks_per_column = segment_rings / num_positions;
        debug_assert_eq!(blocks_per_column * num_positions, segment_rings);
        let row_alignment = (source.one_hot_k / D).max(1);
        let target_tasks = rayon::current_num_threads()
            .saturating_mul(TASKS_PER_RAYON_WORKER)
            .min(num_positions)
            .max(1);
        let thread_balanced_chunk = num_positions
            .div_ceil(target_tasks)
            .next_multiple_of(row_alignment);
        let cache_sized_chunk = (DECOMPOSE_POSITION_WORKING_SET_TARGET
            / std::mem::size_of::<[i32; D]>())
        .max(row_alignment)
        .next_multiple_of(row_alignment);
        let position_chunk = thread_balanced_chunk
            .min(cache_sized_chunk)
            .min(num_positions);
        let position_tasks = num_positions.div_ceil(position_chunk);
        let use_local_dense_rotations = D == 128
            && source.one_hot_k == 256
            && matches!(&rotations, PreparedRotations::Compact(_));
        let local_rotation_rows = if use_local_dense_rotations {
            source.rows.num_columns().checked_mul(D).ok_or_else(|| {
                AkitaError::InvalidInput("local decompose rotation table size overflow".to_string())
            })?
        } else {
            0
        };
        let local_rotation_bytes = local_rotation_rows * std::mem::size_of::<[i16; D]>();
        let _compress_span = tracing::info_span!(
            "trace_onehot_decompose_accumulate",
            mode = "position_parallel",
            num_blocks,
            blocks_per_column,
            position_tasks,
            position_chunk,
            position_working_set_bytes = position_chunk * std::mem::size_of::<[i32; D]>(),
            dense_rotations = rotations.is_dense(),
            local_dense_rotations = use_local_dense_rotations,
            local_rotation_bytes,
        )
        .entered();
        let mut compressed = vec![[0i32; D]; num_positions];
        compressed
            .par_chunks_mut(position_chunk)
            .enumerate()
            .try_for_each(|(position_task, compressed)| {
                let position_start = position_task * position_chunk;
                let position_end = position_start + compressed.len();
                let mut local_rotations =
                    use_local_dense_rotations.then(|| vec![[0i16; D]; local_rotation_rows]);
                for trace_block in 0..blocks_per_column {
                    if let Some(local_rotations) = local_rotations.as_mut() {
                        let PreparedRotations::Compact(challenges) = &rotations else {
                            unreachable!("local dense rotations require compact challenges");
                        };
                        for column in 0..source.rows.num_columns() {
                            let prepared_block = trace_block * source.rows.num_columns() + column;
                            fill_compact_rotation_table(
                                &mut local_rotations[column * D..][..D],
                                &challenges[prepared_block],
                            );
                        }
                    }
                    let ring_start = trace_block * num_positions + position_start;
                    let ring_end = trace_block * num_positions + position_end;
                    if source.one_hot_k < D {
                        let num_columns = source.rows.num_columns();
                        let rows_per_ring = D / source.one_hot_k;
                        let mut coefficients = Vec::with_capacity(rows_per_ring);
                        visit_segment_ring_row_range::<D>(
                            source,
                            ring_start,
                            ring_end,
                            |ring, selected_rows, committed_zero_masks| {
                                let position = ring - trace_block * num_positions;
                                let dst = &mut compressed[position - position_start];
                                if rows_per_ring <= 4 {
                                    for column in 0..num_columns {
                                        let mut fixed_coefficients = [0usize; 4];
                                        let mut count = 0;
                                        for (row_offset, (row_indices, &committed_zero_mask)) in
                                            selected_rows
                                                .chunks_exact(num_columns)
                                                .zip(committed_zero_masks)
                                                .enumerate()
                                        {
                                            let hot = row_indices[column];
                                            if row_is_committed(hot, committed_zero_mask, column) {
                                                fixed_coefficients[count] = row_offset
                                                    * source.one_hot_k
                                                    + usize::from(hot);
                                                count += 1;
                                            }
                                        }
                                        let prepared_block = trace_block * num_columns + column;
                                        add_rotated_rows(
                                            dst,
                                            &rotations,
                                            prepared_block,
                                            &fixed_coefficients[..count],
                                        );
                                    }
                                } else {
                                    for column in 0..num_columns {
                                        coefficients.clear();
                                        for (row_offset, (row_indices, &committed_zero_mask)) in
                                            selected_rows
                                                .chunks_exact(num_columns)
                                                .zip(committed_zero_masks)
                                                .enumerate()
                                        {
                                            let hot = row_indices[column];
                                            if row_is_committed(hot, committed_zero_mask, column) {
                                                coefficients.push(
                                                    row_offset * source.one_hot_k
                                                        + usize::from(hot),
                                                );
                                            }
                                        }
                                        let prepared_block = trace_block * num_columns + column;
                                        add_rotated_rows(
                                            dst,
                                            &rotations,
                                            prepared_block,
                                            &coefficients,
                                        );
                                    }
                                }
                            },
                        )?;
                    } else if let Some(local_rotations) = local_rotations.as_ref() {
                        visit_segment_ring_range::<D>(
                            source,
                            ring_start,
                            ring_end,
                            |ring, contributions| {
                                let position = ring - trace_block * num_positions;
                                add_rotated_dense_contributions(
                                    &mut compressed[position - position_start],
                                    local_rotations,
                                    contributions,
                                    |column, coefficient| column * D + coefficient,
                                );
                            },
                        )?;
                    } else {
                        visit_segment_ring_range::<D>(
                            source,
                            ring_start,
                            ring_end,
                            |ring, contributions| {
                                let position = ring - trace_block * num_positions;
                                add_rotated_contributions(
                                    &mut compressed[position - position_start],
                                    contributions,
                                    &rotations,
                                    trace_block,
                                    source.rows.num_columns(),
                                );
                            },
                        )?;
                    }
                }
                Ok::<_, AkitaError>(())
            })?;
        compressed
    } else {
        let _compress_span = tracing::info_span!(
            "trace_onehot_decompose_accumulate",
            mode = "flat",
            num_blocks,
            segment_rings,
            dense_rotations = rotations.is_dense(),
        )
        .entered();
        let mut compressed = vec![[0i32; D]; num_positions];
        visit_segment_ring_range::<D>(source, 0, segment_rings, |ring, contributions| {
            for &(column, coefficient) in contributions {
                let global_ring = column * segment_rings + ring;
                let block = global_ring / num_positions;
                add_rotated(
                    &mut compressed[global_ring % num_positions],
                    &rotations,
                    block,
                    coefficient,
                );
            }
        })?;
        compressed
    };
    let _expand_span = tracing::info_span!(
        "trace_onehot_decompose_expand_digits",
        num_positions,
        num_digits,
    )
    .entered();
    let expanded = if num_digits == 1 {
        compressed
    } else {
        let mut expanded = Vec::with_capacity(num_positions.saturating_mul(num_digits));
        for coeffs in compressed {
            expanded.push(coeffs);
            expanded.extend((1..num_digits).map(|_| [0i32; D]));
        }
        expanded
    };
    drop(_expand_span);
    let modulus = (-AkitaField::one()).to_canonical_u128() + 1;
    let _witness_span = tracing::info_span!(
        "trace_onehot_decompose_build_witness",
        num_positions,
        num_digits,
    )
    .entered();
    Ok(build_decompose_fold_witness::<AkitaField, D>(
        expanded, modulus,
    ))
}

pub(super) fn decompose_fold_packed<const D: usize>(
    source: &TracePackedOneHot,
    challenges: &[SparseChallenge],
    num_positions: usize,
    num_digits: usize,
) -> Result<DecomposeFoldWitness<AkitaField>, AkitaError> {
    decompose_fold_packed_with_mode::<D>(
        source,
        challenges,
        num_positions,
        num_digits,
        DecomposeRotationMode::from_env()?,
    )
}
