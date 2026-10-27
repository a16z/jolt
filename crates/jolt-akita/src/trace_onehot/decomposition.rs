use std::env::VarError;
use std::ops::{AddAssign, Range};

use akita_challenges::SparseChallenge;
use akita_error::AkitaError;
use akita_pcs::custom_source::{fill_rotated_challenge, DecomposeFoldWitness};
use rayon::prelude::*;
use tracing::field::Empty;

use super::source::TraceOneHotColumn;
use super::traversal::{
    row_is_committed, validate_block_geometry, visit_segment_ring_range,
    visit_segment_ring_row_batches, visit_segment_ring_row_range,
};
use super::{
    DECOMPOSE_POSITION_WORKING_SET_TARGET, ROTATED_CHALLENGE_TABLE_BUDGET, TASKS_PER_RAYON_WORKER,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DecomposeRotationMode {
    Auto,
    Compact,
    Dense,
    Sparse,
}

impl DecomposeRotationMode {
    pub(super) fn from_env() -> Result<Self, AkitaError> {
        match std::env::var("JOLT_AKITA_DECOMPOSE_MODE").as_deref() {
            Ok("compact") => Ok(Self::Compact),
            Ok("dense") => Ok(Self::Dense),
            Ok("sparse") => Ok(Self::Sparse),
            Ok("auto") | Err(VarError::NotPresent) => Ok(Self::Auto),
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

    #[inline(always)]
    fn accumulate_rows(
        &self,
        source: &TraceOneHotColumn,
        rings: Range<usize>,
        destination: &mut [[i32; D]],
        block_index: impl Fn(usize) -> usize,
    ) -> Result<(), AkitaError> {
        match self {
            Self::Dense(rotated) => {
                accumulate_dense_row_range(source, rings, destination, rotated, block_index)
            }
            Self::Compact(challenges) => accumulate_row_range::<D>(source, rings, destination, {
                #[inline(always)]
                |dst, column, coefficients| {
                    let challenge = &challenges[block_index(column)];
                    for &coefficient in coefficients {
                        add_rotated_compact(dst, challenge, coefficient);
                    }
                }
            }),
            Self::Sparse(challenges) => accumulate_row_range::<D>(source, rings, destination, {
                #[inline(always)]
                |dst, column, coefficients| {
                    let challenge = &challenges[block_index(column)];
                    for &coefficient in coefficients {
                        add_rotated_sparse(dst, challenge, coefficient);
                    }
                }
            }),
        }
    }

    #[inline(always)]
    fn accumulate_contributions(
        &self,
        source: &TraceOneHotColumn,
        rings: Range<usize>,
        destination: &mut [[i32; D]],
        trace_block: usize,
    ) -> Result<(), AkitaError> {
        let first = trace_block * source.num_columns;
        match self {
            Self::Dense(rotated) => accumulate_dense_ring_rows(
                source,
                rings,
                destination,
                &rotated[first * D..][..source.num_columns * D],
            ),
            Self::Compact(challenges) => {
                let ring_start = rings.start;
                visit_segment_ring_range::<D>(source, rings.start, rings.end, {
                    #[inline(always)]
                    |ring, contributions| {
                        for &(column, coefficient) in contributions {
                            add_rotated_compact(
                                &mut destination[ring - ring_start],
                                &challenges[first + column],
                                coefficient,
                            );
                        }
                    }
                })
            }
            Self::Sparse(challenges) => {
                let ring_start = rings.start;
                let ring_end = rings.end;
                let rings_per_row = source.one_hot_k / D;
                visit_segment_ring_row_batches::<D>(
                    source,
                    ring_start,
                    ring_end,
                    |batch_start, selected_rows, committed_zero_masks| {
                        for (row_offset, (selected_rows, &committed_zero_mask)) in selected_rows
                            .chunks_exact(source.num_columns)
                            .zip(committed_zero_masks)
                            .enumerate()
                        {
                            let row_ring = (batch_start + row_offset) * rings_per_row;
                            for (column, &hot) in selected_rows.iter().enumerate() {
                                if row_is_committed(hot, committed_zero_mask, column) {
                                    let hot = usize::from(hot);
                                    let ring = row_ring + hot / D;
                                    if ring_start <= ring && ring < ring_end {
                                        add_rotated_sparse(
                                            &mut destination[ring - ring_start],
                                            &challenges[first + column],
                                            hot % D,
                                        );
                                    }
                                }
                            }
                        }
                    },
                )
            }
        }
    }
}

enum DensePositionTask<'a, const D: usize> {
    Wide(&'a mut [[i32; D]]),
    Narrow {
        destination: &'a mut [[i32; D]],
        partials: Vec<[i16; D]>,
    },
}

impl<'a, const D: usize> DensePositionTask<'a, D> {
    fn new(destination: &'a mut [[i32; D]], narrow: bool) -> Self {
        if narrow {
            let partials = vec![[0i16; D]; destination.len()];
            Self::Narrow {
                destination,
                partials,
            }
        } else {
            Self::Wide(destination)
        }
    }

    fn accumulate_batch(
        &mut self,
        source: &TraceOneHotColumn,
        first_ring: usize,
        num_positions: usize,
        rotated: &[[i16; D]],
    ) -> Result<(), AkitaError> {
        let positions = match self {
            Self::Wide(destination) | Self::Narrow { destination, .. } => destination.len(),
        };
        match self {
            Self::Wide(destination) => {
                for (offset, tables) in rotated.chunks_exact(source.num_columns * D).enumerate() {
                    let start = first_ring + offset * num_positions;
                    accumulate_dense_ring_rows(
                        source,
                        start..start + positions,
                        destination,
                        tables,
                    )?;
                }
            }
            Self::Narrow { partials, .. } => {
                for (offset, tables) in rotated.chunks_exact(source.num_columns * D).enumerate() {
                    let start = first_ring + offset * num_positions;
                    accumulate_dense_ring_rows(source, start..start + positions, partials, tables)?;
                }
            }
        }
        Ok(())
    }

    fn flush(&mut self) {
        if let Self::Narrow {
            destination,
            partials,
        } = self
        {
            for (destination, partial) in destination.iter_mut().zip(partials) {
                if partial.iter().any(|&coefficient| coefficient != 0) {
                    add_rotated_dense(destination, partial);
                    partial.fill(0);
                }
            }
        }
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
    let use_compact = mode == DecomposeRotationMode::Compact
        || (mode == DecomposeRotationMode::Auto
            && (D == 128
                || ((D >= 256 || (D == 64 && dense_bytes > ROTATED_CHALLENGE_TABLE_BUDGET))
                    && challenges
                        .iter()
                        .all(|challenge| challenge.positions.len() >= D / 4))));
    if use_compact {
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
fn add_rotated_dense<const D: usize, Accumulator: From<i16> + AddAssign>(
    dst: &mut [Accumulator; D],
    rotated: &[i16; D],
) {
    for (dst, &value) in dst.iter_mut().zip(rotated) {
        *dst += Accumulator::from(value);
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
fn add_rotated_dense_tables<const D: usize, const N: usize, Accumulator: From<i16> + AddAssign>(
    dst: &mut [Accumulator; D],
    tables: [&[i16; D]; N],
) {
    const { assert!(N <= 8) };
    // Rotated i8 challenges have magnitude <=128. Eight entries fit in i16,
    // letting SIMD sum narrow lanes before widening once for accumulation.
    for coefficient in 0..D {
        let mut sum = 0i16;
        for table in tables {
            sum += table[coefficient];
        }
        dst[coefficient] += Accumulator::from(sum);
    }
}

#[inline(always)]
fn add_rotated_dense_tail<const D: usize, Accumulator: From<i16> + AddAssign>(
    dst: &mut [Accumulator; D],
    tables: &[&[i16; D]],
) {
    match tables {
        [] => {}
        [t0] => add_rotated_dense(dst, t0),
        [t0, t1] => add_rotated_dense_tables(dst, [*t0, *t1]),
        [t0, t1, t2] => add_rotated_dense_tables(dst, [*t0, *t1, *t2]),
        [t0, t1, t2, t3] => {
            add_rotated_dense_tables(dst, [*t0, *t1, *t2, *t3]);
        }
        [t0, t1, t2, t3, t4] => {
            add_rotated_dense_tables(dst, [*t0, *t1, *t2, *t3, *t4]);
        }
        [t0, t1, t2, t3, t4, t5] => add_rotated_dense_tables(dst, [*t0, *t1, *t2, *t3, *t4, *t5]),
        [t0, t1, t2, t3, t4, t5, t6] => {
            add_rotated_dense_tables(dst, [*t0, *t1, *t2, *t3, *t4, *t5, *t6]);
        }
        _ => unreachable!("eight-entry batches leave at most seven contributions"),
    }
}

#[inline(always)]
fn accumulate_dense_ring_rows<const D: usize, Accumulator: From<i16> + AddAssign>(
    source: &TraceOneHotColumn,
    rings: Range<usize>,
    destination: &mut [[Accumulator; D]],
    rotated: &[[i16; D]],
) -> Result<(), AkitaError> {
    // Byte row indices bound the number of rings per row. Keep the common
    // D>=64 bucket buffers on the stack with a compile-time capacity.
    if D >= 64 {
        accumulate_dense_ring_rows_with_capacity::<D, 4, Accumulator>(
            source,
            rings,
            destination,
            rotated,
        )
    } else {
        accumulate_dense_ring_rows_with_capacity::<D, 256, Accumulator>(
            source,
            rings,
            destination,
            rotated,
        )
    }
}

#[inline(always)]
fn accumulate_dense_ring_rows_with_capacity<
    const D: usize,
    const C: usize,
    Accumulator: From<i16> + AddAssign,
>(
    source: &TraceOneHotColumn,
    rings: Range<usize>,
    destination: &mut [[Accumulator; D]],
    rotated: &[[i16; D]],
) -> Result<(), AkitaError> {
    let ring_start = rings.start;
    let ring_end = rings.end;
    let rings_per_row = source.one_hot_k / D;
    let (column_rotations, remainder) = rotated.as_chunks::<D>();
    debug_assert!(remainder.is_empty());
    let mut tables = [[&rotated[0]; 8]; C];
    let mut counts = [0usize; C];
    visit_segment_ring_row_batches::<D>(
        source,
        ring_start,
        ring_end,
        |batch_start, selected_rows, committed_zero_masks| {
            let rotations = column_rotations;
            for (row_offset, (selected_rows, &committed_zero_mask)) in selected_rows
                .chunks_exact(source.num_columns)
                .zip(committed_zero_masks)
                .enumerate()
            {
                counts[..rings_per_row].fill(0);
                let row_ring = (batch_start + row_offset) * rings_per_row;
                for (column_offset, (&hot, rotations)) in
                    selected_rows.iter().zip(rotations).enumerate()
                {
                    let column = column_offset;
                    if !row_is_committed(hot, committed_zero_mask, column) {
                        continue;
                    }
                    let hot = usize::from(hot);
                    let offset = hot / D;
                    let ring = row_ring + offset;
                    if ring < ring_start || ring >= ring_end {
                        continue;
                    }
                    let count = &mut counts[offset];
                    let batch = &mut tables[offset];
                    batch[*count] = &rotations[hot % D];
                    *count += 1;
                    if *count == batch.len() {
                        add_rotated_dense_tables(&mut destination[ring - ring_start], *batch);
                        *count = 0;
                    }
                }
                for (offset, (&count, batch)) in
                    counts[..rings_per_row].iter().zip(&tables).enumerate()
                {
                    if count != 0 {
                        add_rotated_dense_tail(
                            &mut destination[row_ring + offset - ring_start],
                            &batch[..count],
                        );
                    }
                }
            }
        },
    )
}

#[inline(always)]
fn accumulate_dense_row_range<const D: usize>(
    source: &TraceOneHotColumn,
    rings: Range<usize>,
    destination: &mut [[i32; D]],
    rotations: &[[i16; D]],
    block_index: impl Fn(usize) -> usize,
) -> Result<(), AkitaError> {
    let num_columns = source.num_columns;
    let ring_start = rings.start;
    let first_rotation = rotations
        .first()
        .ok_or_else(|| AkitaError::InvalidInput("empty dense decompose rotation table".into()))?;
    visit_segment_ring_row_range::<D>(source, rings.start, rings.end, {
        #[inline(always)]
        |ring, selected_rows, committed_zero_masks| {
            let position = ring - ring_start;
            let dst = &mut destination[position];
            // Combine columns sharing a destination to fill eight-way additions.
            let mut tables = [first_rotation; 8];
            let mut count = 0;
            for column in 0..num_columns {
                let base = block_index(column) * D;
                let column_rotations = &rotations[base..][..D];
                for (row_offset, &mask) in committed_zero_masks.iter().enumerate() {
                    let hot = selected_rows[row_offset * num_columns + column];
                    if row_is_committed(hot, mask, column) {
                        tables[count] =
                            &column_rotations[row_offset * source.one_hot_k + usize::from(hot)];
                        count += 1;
                        if count == tables.len() {
                            add_rotated_dense_tables(dst, tables);
                            count = 0;
                        }
                    }
                }
            }
            add_rotated_dense_tail(dst, &tables[..count]);
        }
    })
}

#[inline(always)]
fn accumulate_row_range<const D: usize>(
    source: &TraceOneHotColumn,
    rings: Range<usize>,
    destination: &mut [[i32; D]],
    add_rows: impl Fn(&mut [i32; D], usize, &[usize]),
) -> Result<(), AkitaError> {
    // Preserve a compile-time bound for small row batches, uniformly across chunk counts.
    if D / source.one_hot_k <= 4 {
        accumulate_row_range_with_capacity::<D, 4>(source, rings, destination, add_rows)
    } else {
        accumulate_row_range_with_capacity::<D, D>(source, rings, destination, add_rows)
    }
}

#[inline(always)]
fn accumulate_row_range_with_capacity<const D: usize, const C: usize>(
    source: &TraceOneHotColumn,
    rings: Range<usize>,
    destination: &mut [[i32; D]],
    add_rows: impl Fn(&mut [i32; D], usize, &[usize]),
) -> Result<(), AkitaError> {
    let num_columns = source.num_columns;
    let mut coefficients = [0usize; C];
    let ring_start = rings.start;
    visit_segment_ring_row_range::<D>(
        source,
        rings.start,
        rings.end,
        |ring, selected_rows, committed_zero_masks| {
            let position = ring - ring_start;
            let dst = &mut destination[position];
            for column in 0..num_columns {
                let mut count = 0;
                for (row_offset, &committed_zero_mask) in committed_zero_masks.iter().enumerate() {
                    let hot = selected_rows[row_offset * num_columns + column];
                    if row_is_committed(hot, committed_zero_mask, column) {
                        coefficients[count] = row_offset * source.one_hot_k + usize::from(hot);
                        count += 1;
                    }
                }
                add_rows(dst, column, &coefficients[..count]);
            }
        },
    )
}

fn positions_per_task<const D: usize>(
    positions: usize,
    row_alignment: usize,
    working_set_target: usize,
) -> usize {
    let target_tasks = rayon::current_num_threads()
        .saturating_mul(TASKS_PER_RAYON_WORKER)
        .min(positions)
        .max(1);
    let thread_balanced_chunk = positions
        .div_ceil(target_tasks)
        .next_multiple_of(row_alignment);
    let bytes_per_position = std::mem::size_of::<[i32; D]>();
    let cache_sized_chunk = (working_set_target / bytes_per_position)
        .max(row_alignment)
        .next_multiple_of(row_alignment);
    thread_balanced_chunk.min(cache_sized_chunk).min(positions)
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

pub(super) fn decompose_fold_columns_with_mode<const D: usize>(
    source: &TraceOneHotColumn,
    challenges: &[SparseChallenge],
    chunk_ranges: &[Range<usize>],
    num_positions: usize,
    num_digits: usize,
    rotation_mode: DecomposeRotationMode,
) -> Result<Vec<DecomposeFoldWitness>, AkitaError> {
    let num_chunks = chunk_ranges.len();
    let _span = tracing::info_span!(
        "TraceOneHotColumn::decompose_fold_batch",
        ring_dimension = D,
        rows = source.num_rows,
        columns = source.num_columns,
        num_chunks,
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
        validate_block_geometry(segment_rings, source.num_columns, num_positions)?;
    if challenges.len() != num_blocks {
        return Err(AkitaError::InvalidSize {
            expected: num_blocks,
            actual: challenges.len(),
        });
    }
    for challenge in challenges {
        challenge.validate::<D>()?;
    }
    let blocks_per_poly = num_blocks / source.num_columns;
    let expected_ranges = akita_params::dyadic_block_ranges(blocks_per_poly, num_chunks)?;
    if chunk_ranges != expected_ranges {
        return Err(AkitaError::InvalidInput(
            "noncanonical fold chunk ranges".into(),
        ));
    }
    let blocks_per_column = (segment_rings >= num_positions).then(|| segment_rings / num_positions);
    let rotation_table_bytes = num_blocks
        .saturating_mul(D)
        .saturating_mul(std::mem::size_of::<[i16; D]>());
    let rotation_span = tracing::info_span!(
        "trace_onehot_decompose_prepare_rotations",
        challenge_blocks = challenges.len(),
        rotation_table_bytes,
        table_budget_bytes = ROTATED_CHALLENGE_TABLE_BUDGET,
        requested_mode = ?rotation_mode,
        dense = Empty,
    );
    let rotation_guard = rotation_span.enter();
    let rotations = prepare_rotations::<D>(
        challenges,
        blocks_per_column,
        source.num_columns,
        rotation_mode,
    )?;
    let _ = rotation_span.record("dense", rotations.is_dense());
    drop(rotation_guard);
    let mut compressed = (0..num_chunks)
        .map(|_| vec![[0i32; D]; num_positions])
        .collect::<Vec<_>>();
    if segment_rings >= num_positions {
        let blocks_per_column = segment_rings / num_positions;
        debug_assert_eq!(blocks_per_column * num_positions, segment_rings);
        let position_chunk = positions_per_task::<D>(
            num_positions,
            (source.one_hot_k / D).max(1),
            DECOMPOSE_POSITION_WORKING_SET_TARGET,
        );
        let position_tasks = num_positions.div_ceil(position_chunk);
        let use_block_dense_rotations = D == 128
            && source.one_hot_k == 256
            && matches!(&rotations, PreparedRotations::Compact(_));
        let _compress_span = tracing::info_span!(
            "trace_onehot_decompose_accumulate",
            mode = "position_parallel",
            num_blocks,
            blocks_per_column,
            position_tasks,
            position_chunk,
            position_working_set_bytes = position_chunk * std::mem::size_of::<[i32; D]>(),
            dense_rotations = rotations.is_dense(),
            local_dense_rotations = use_block_dense_rotations,
        )
        .entered();
        if use_block_dense_rotations {
            let PreparedRotations::Compact(challenges) = &rotations else {
                unreachable!("block dense rotations require compact challenges");
            };
            const BLOCK_ROTATION_WORKING_SET_TARGET: usize = 1 << 22;
            let block_rotation_rows = source.num_columns * D;
            let block_rotation_bytes = block_rotation_rows * std::mem::size_of::<[i16; D]>();
            // K>=D contributes at most one rotated i8 value per column to a position.
            let coefficient_bound = source.num_columns * usize::from(i8::MIN.unsigned_abs());
            let blocks_per_batch = (BLOCK_ROTATION_WORKING_SET_TARGET / block_rotation_bytes)
                .max(1)
                .min(i16::MAX as usize / coefficient_bound)
                .min(blocks_per_column);
            let mut tasks = compressed
                .iter_mut()
                .zip(chunk_ranges)
                .flat_map(|(destination, blocks)| {
                    destination.chunks_mut(position_chunk).map(|destination| {
                        // Narrow partials require enough repeated updates to amortize widening.
                        DensePositionTask::new(destination, blocks.len() >= 4)
                    })
                })
                .collect::<Vec<_>>();
            let mut partial_bounds = vec![0usize; num_chunks];
            let mut block_rotations = vec![[0i16; D]; block_rotation_rows * blocks_per_batch];
            for batch_start in (0..blocks_per_column).step_by(blocks_per_batch) {
                let batch_end = (batch_start + blocks_per_batch).min(blocks_per_column);
                let block_rotations =
                    &mut block_rotations[..block_rotation_rows * (batch_end - batch_start)];
                block_rotations
                    .par_chunks_mut(D)
                    .enumerate()
                    .for_each(|(index, table)| {
                        fill_compact_rotation_table(
                            table,
                            &challenges[batch_start * source.num_columns + index],
                        );
                    });
                let batch_bounds = chunk_ranges
                    .iter()
                    .map(|blocks| {
                        let first = blocks.start.max(batch_start);
                        let end = blocks.end.min(batch_end);
                        (first..end)
                            .map(|block| {
                                challenges[block * source.num_columns..][..source.num_columns]
                                    .iter()
                                    .map(|challenge| {
                                        challenge
                                            .iter()
                                            .map(|value| usize::from(value.unsigned_abs()))
                                            .max()
                                            .unwrap_or(0)
                                    })
                                    .sum::<usize>()
                            })
                            .sum::<usize>()
                    })
                    .collect::<Vec<_>>();
                for (chunk, (partial, batch)) in
                    partial_bounds.iter_mut().zip(batch_bounds).enumerate()
                {
                    if *partial + batch > i16::MAX as usize {
                        tasks[chunk * position_tasks..(chunk + 1) * position_tasks]
                            .par_iter_mut()
                            .for_each(DensePositionTask::flush);
                        *partial = 0;
                    }
                    *partial += batch;
                }
                // Canonical ranges are ordered. Dispatch only the chunk tasks
                // intersecting this table batch, including across a chunk boundary.
                let first_chunk = chunk_ranges.partition_point(|blocks| blocks.end <= batch_start);
                let end_chunk = chunk_ranges.partition_point(|blocks| blocks.start < batch_end);
                let first_task = first_chunk * position_tasks;
                tasks[first_task..end_chunk * position_tasks]
                    .par_iter_mut()
                    .enumerate()
                    .try_for_each(|(offset, task)| {
                        let index = first_task + offset;
                        let blocks = &chunk_ranges[index / position_tasks];
                        let first = blocks.start.max(batch_start);
                        let end = blocks.end.min(batch_end);
                        if first < end {
                            let first_ring =
                                first * num_positions + index % position_tasks * position_chunk;
                            task.accumulate_batch(
                                source,
                                first_ring,
                                num_positions,
                                &block_rotations[(first - batch_start) * block_rotation_rows
                                    ..(end - batch_start) * block_rotation_rows],
                            )?;
                        }
                        Ok::<_, AkitaError>(())
                    })?;
            }
            tasks.par_iter_mut().for_each(DensePositionTask::flush);
        } else {
            // Each chunk owns the same block range of every column. Tasks touch one
            // contiguous output buffer and read only that range of shared trace rows.
            compressed
                .par_iter_mut()
                .zip(chunk_ranges)
                .try_for_each(|(destination, blocks)| {
                    destination
                        .par_chunks_mut(position_chunk)
                        .enumerate()
                        .try_for_each(|(task, destination)| {
                            let position_start = task * position_chunk;
                            let position_end = position_start + destination.len();
                            for trace_block in blocks.clone() {
                                let rings = trace_block * num_positions + position_start
                                    ..trace_block * num_positions + position_end;
                                if source.one_hot_k < D {
                                    let first = trace_block * source.num_columns;
                                    rotations.accumulate_rows(
                                        source,
                                        rings,
                                        destination,
                                        |column| first + column,
                                    )?;
                                } else {
                                    rotations.accumulate_contributions(
                                        source,
                                        rings,
                                        destination,
                                        trace_block,
                                    )?;
                                }
                            }
                            Ok::<_, AkitaError>(())
                        })
                })?;
        }
    } else {
        let position_chunk = positions_per_task::<D>(
            segment_rings,
            (source.one_hot_k / D).max(1),
            DECOMPOSE_POSITION_WORKING_SET_TARGET,
        );
        let _compress_span = tracing::info_span!(
            "trace_onehot_decompose_accumulate",
            mode = "flat_position_parallel",
            num_blocks,
            segment_rings,
            position_chunk,
            dense_rotations = rotations.is_dense(),
        )
        .entered();
        // A short native polynomial has one live block; the dyadic partition
        // may leave other chunks empty. No selector slots are part of this layout.
        compressed
            .par_iter_mut()
            .zip(chunk_ranges)
            .try_for_each(|(destination, blocks)| {
                if blocks.is_empty() {
                    return Ok(());
                }
                destination[..segment_rings]
                    .par_chunks_mut(position_chunk)
                    .enumerate()
                    .try_for_each(|(task, destination)| {
                        let ring_start = task * position_chunk;
                        let rings = ring_start..ring_start + destination.len();
                        if source.one_hot_k < D {
                            rotations
                                .accumulate_rows(source, rings, destination, |column| column)?;
                        } else {
                            rotations.accumulate_contributions(source, rings, destination, 0)?;
                        }
                        Ok::<_, AkitaError>(())
                    })
            })?;
    }
    let _expand_span = tracing::info_span!(
        "trace_onehot_decompose_expand_digits",
        num_positions,
        num_digits,
    )
    .entered();
    let expanded = if num_digits == 1 {
        compressed
    } else {
        compressed
            .into_par_iter()
            .map(|compressed| {
                let mut expanded = Vec::with_capacity(num_positions.saturating_mul(num_digits));
                for coeffs in compressed {
                    expanded.push(coeffs);
                    expanded.extend((1..num_digits).map(|_| [0i32; D]));
                }
                expanded
            })
            .collect()
    };
    drop(_expand_span);
    let _witness_span = tracing::info_span!(
        "trace_onehot_decompose_build_witness",
        num_positions,
        num_digits,
    )
    .entered();
    Ok(expanded
        .into_par_iter()
        .map(DecomposeFoldWitness::from_centered_rows::<D>)
        .collect())
}
