//! Streaming kernels for Jolt's prefix-packed trace one-hot polynomial.
//!
//! A trace row produces every semantic hot lane. Jolt's kernels consume that
//! row-major source directly and lay the columns out as consecutive `K * T`
//! segments inside one physical polynomial. Padding selector slots are zero.

use std::fmt;
use std::sync::Arc;

use akita_algebra::ring::WideCyclotomicRing;
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_field::unreduced::HasWide;
use akita_field::{AkitaError, ExtField, FromPrimitiveInt, MulBaseUnreduced};
use akita_prover::backend::poly_helpers::build_decompose_fold_witness;
use akita_prover::compute::{
    CommitInnerPlan, DecomposeFoldBatchPlan, DecomposeFoldPlan, OpeningBatchKernel,
    OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan, RootCommitKernel, TensorPackedWitness,
    TensorProjectionBatchKernel, TensorProjectionKernel,
};
use akita_prover::kernels::linear::decompose_commit_blocks_into;
use akita_prover::{
    BatchDecomposeFoldOutcome, CommitInnerWitness, ComputeBackendSetup, CpuBackend,
    RootCommitSource, RootOpeningSource, RootPolyMeta, RootPolyShape, RootTensorProjectionPoly,
    RootTensorSource,
};
use akita_types::FpExtEncoding;
use rayon::prelude::*;

use crate::AkitaField;

const NO_HOT_LANE: u16 = u16::MAX;
const MAX_WIDE_ACCUMULATIONS: usize = 1 << 15;
const TASKS_PER_RAYON_WORKER: usize = 4;

/// Row-major source for the semantic columns packed into `OneHotTrace`.
///
/// `fill_row` must overwrite all of `hot_lanes`. Use [`no_hot_lane`] when a
/// semantic column has no nonzero entry in that row.
pub trait TraceOneHotRows: Send + Sync + 'static {
    fn num_rows(&self) -> usize;
    fn num_columns(&self) -> usize;
    fn fill_row(&self, row: usize, hot_lanes: &mut [u16]);
}

/// Sentinel written by [`TraceOneHotRows::fill_row`] for an empty one-hot row.
#[must_use]
pub const fn no_hot_lane() -> u16 {
    NO_HOT_LANE
}

/// One physical one-hot polynomial containing all trace-derived semantic
/// columns and zero padding up to a protocol-fixed selector capacity.
#[derive(Clone)]
pub struct TracePackedOneHot {
    rows: Arc<dyn TraceOneHotRows>,
    one_hot_k: usize,
    column_capacity: usize,
    num_vars: usize,
    construction_ring_elems: usize,
}

impl fmt::Debug for TracePackedOneHot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TracePackedOneHot")
            .field("one_hot_k", &self.one_hot_k)
            .field("num_columns", &self.rows.num_columns())
            .field("column_capacity", &self.column_capacity)
            .field("num_vars", &self.num_vars)
            .finish_non_exhaustive()
    }
}

impl TracePackedOneHot {
    /// Constructs one prefix-packed source.
    ///
    /// `construction_ring_d` is metadata matching the configured Akita
    /// commitment dimension. Kernel views remain const-generic over `D`.
    pub fn new(
        one_hot_k: usize,
        construction_ring_d: usize,
        column_capacity: usize,
        rows: Arc<dyn TraceOneHotRows>,
    ) -> Result<Self, AkitaError> {
        if !one_hot_k.is_power_of_two() || one_hot_k > usize::from(u16::MAX) {
            return Err(AkitaError::InvalidInput(format!(
                "trace one-hot K={one_hot_k} must be a power of two fitting u16 lanes below the empty sentinel"
            )));
        }
        if construction_ring_d == 0 || !construction_ring_d.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "trace one-hot construction D={construction_ring_d} must be a power of two"
            )));
        }
        if !column_capacity.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "trace one-hot column capacity {column_capacity} must be a power of two"
            )));
        }
        let num_columns = rows.num_columns();
        if num_columns == 0 || num_columns > column_capacity {
            return Err(AkitaError::InvalidInput(format!(
                "trace one-hot has {num_columns} semantic columns for capacity {column_capacity}"
            )));
        }
        let total_field_elems = rows
            .num_rows()
            .checked_mul(one_hot_k)
            .and_then(|segment| segment.checked_mul(column_capacity))
            .ok_or_else(|| {
                AkitaError::InvalidInput("trace one-hot packed domain overflow".to_string())
            })?;
        if !total_field_elems.is_power_of_two()
            || !total_field_elems.is_multiple_of(construction_ring_d)
        {
            return Err(AkitaError::InvalidInput(format!(
                "trace one-hot packed domain {total_field_elems} must be a power of two divisible by construction D={construction_ring_d}"
            )));
        }
        Ok(Self {
            rows,
            one_hot_k,
            column_capacity,
            num_vars: total_field_elems.trailing_zeros() as usize,
            construction_ring_elems: total_field_elems / construction_ring_d,
        })
    }

    fn total_field_elems(&self) -> usize {
        1usize << self.num_vars
    }

    fn segment_ring_elems<const D: usize>(&self) -> Result<usize, AkitaError> {
        validate_dimension::<D>(self.one_hot_k)?;
        let segment_field_elems = self
            .rows
            .num_rows()
            .checked_mul(self.one_hot_k)
            .ok_or_else(|| {
                AkitaError::InvalidInput("trace one-hot segment ring count overflow".to_string())
            })?;
        if !segment_field_elems.is_multiple_of(D) {
            return Err(AkitaError::InvalidInput(format!(
                "trace one-hot semantic segment {segment_field_elems} is not ring-aligned at D={D}"
            )));
        }
        Ok(segment_field_elems / D)
    }
}

#[derive(Clone, Copy)]
pub struct TracePackedOneHotView<'a, const D: usize> {
    source: &'a TracePackedOneHot,
}

#[derive(Clone, Copy)]
pub struct TracePackedOneHotBatchView<'a, const D: usize> {
    sources: &'a [&'a TracePackedOneHot],
}

impl RootPolyMeta<AkitaField> for TracePackedOneHot {
    fn num_ring_elems(&self) -> usize {
        self.construction_ring_elems
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        Some(self.one_hot_k)
    }
}

impl<const D: usize> RootPolyShape<AkitaField, D> for TracePackedOneHot {
    fn num_ring_elems(&self) -> usize {
        self.total_field_elems().div_ceil(D)
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        Some(self.one_hot_k)
    }
}

impl<const D: usize> RootCommitSource<AkitaField, D> for TracePackedOneHot {
    type CommitView<'a>
        = TracePackedOneHotView<'a, D>
    where
        Self: 'a;

    fn commit_view(&self) -> Result<Self::CommitView<'_>, AkitaError> {
        validate_dimension::<D>(self.one_hot_k)?;
        Ok(TracePackedOneHotView { source: self })
    }
}

impl<const D: usize> RootOpeningSource<AkitaField, D> for TracePackedOneHot {
    type OpeningView<'a>
        = TracePackedOneHotView<'a, D>
    where
        Self: 'a;
    type OpeningBatchView<'a>
        = TracePackedOneHotBatchView<'a, D>
    where
        Self: 'a;

    fn opening_view(&self) -> Result<Self::OpeningView<'_>, AkitaError> {
        validate_dimension::<D>(self.one_hot_k)?;
        Ok(TracePackedOneHotView { source: self })
    }

    fn opening_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::OpeningBatchView<'a>, AkitaError> {
        validate_singleton_batch(polys)?;
        validate_dimension::<D>(polys[0].one_hot_k)?;
        Ok(TracePackedOneHotBatchView { sources: polys })
    }
}

impl<const D: usize> RootTensorSource<AkitaField, D> for TracePackedOneHot {
    type TensorView<'a>
        = TracePackedOneHotView<'a, D>
    where
        Self: 'a;
    type TensorBatchView<'a>
        = TracePackedOneHotBatchView<'a, D>
    where
        Self: 'a;

    fn tensor_view(&self) -> Result<Self::TensorView<'_>, AkitaError> {
        validate_dimension::<D>(self.one_hot_k)?;
        Ok(TracePackedOneHotView { source: self })
    }

    fn tensor_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::TensorBatchView<'a>, AkitaError> {
        validate_singleton_batch(polys)?;
        validate_dimension::<D>(polys[0].one_hot_k)?;
        Ok(TracePackedOneHotBatchView { sources: polys })
    }
}

fn validate_singleton_batch(polys: &[&TracePackedOneHot]) -> Result<(), AkitaError> {
    if polys.len() != 1 {
        return Err(AkitaError::InvalidSize {
            expected: 1,
            actual: polys.len(),
        });
    }
    Ok(())
}

fn validate_dimension<const D: usize>(one_hot_k: usize) -> Result<(), AkitaError> {
    if D == 0
        || !D.is_power_of_two()
        || !(one_hot_k.is_multiple_of(D) || D.is_multiple_of(one_hot_k))
    {
        return Err(AkitaError::InvalidInput(format!(
            "trace one-hot K={one_hot_k} and D={D} must be powers of two with one dividing the other"
        )));
    }
    Ok(())
}

/// Visits ring elements within one semantic column segment. Each callback
/// receives the segment-relative ring index and `(column, coefficient)` pairs
/// contributed by the same trace rows.
fn visit_segment_ring_range<const D: usize>(
    source: &TracePackedOneHot,
    ring_start: usize,
    ring_end: usize,
    mut visit: impl FnMut(usize, &[(usize, usize)]),
) -> Result<(), AkitaError> {
    validate_dimension::<D>(source.one_hot_k)?;
    let segment_rings = source.segment_ring_elems::<D>()?;
    if ring_start > ring_end || ring_end > segment_rings {
        return Err(AkitaError::InvalidInput(format!(
            "trace one-hot ring range {ring_start}..{ring_end} exceeds segment size {segment_rings}"
        )));
    }
    let k = source.one_hot_k;
    let num_columns = source.rows.num_columns();
    let mut lanes = vec![NO_HOT_LANE; num_columns];
    if k >= D {
        let rings_per_row = k / D;
        let row_start = ring_start / rings_per_row;
        let row_end = ring_end.div_ceil(rings_per_row);
        let mut buckets = vec![Vec::new(); rings_per_row];
        for row in row_start..row_end.min(source.rows.num_rows()) {
            source.rows.fill_row(row, &mut lanes);
            for bucket in &mut buckets {
                bucket.clear();
            }
            for (column, &hot) in lanes.iter().enumerate() {
                if hot == NO_HOT_LANE {
                    continue;
                }
                let hot = usize::from(hot);
                if hot >= k {
                    return Err(AkitaError::InvalidInput(format!(
                        "trace one-hot lane {hot} is outside K={k}"
                    )));
                }
                buckets[hot / D].push((column, hot % D));
            }
            for (offset, contributions) in buckets.iter().enumerate() {
                let ring = row * rings_per_row + offset;
                if ring_start <= ring && ring < ring_end {
                    visit(ring, contributions);
                }
            }
        }
    } else {
        let rows_per_ring = D / k;
        let mut contributions = Vec::with_capacity(num_columns * rows_per_ring);
        for ring in ring_start..ring_end {
            contributions.clear();
            for row_offset in 0..rows_per_ring {
                let row = ring * rows_per_ring + row_offset;
                if row >= source.rows.num_rows() {
                    break;
                }
                source.rows.fill_row(row, &mut lanes);
                for (column, &hot) in lanes.iter().enumerate() {
                    if hot == NO_HOT_LANE {
                        continue;
                    }
                    let hot = usize::from(hot);
                    if hot >= k {
                        return Err(AkitaError::InvalidInput(format!(
                            "trace one-hot lane {hot} is outside K={k}"
                        )));
                    }
                    contributions.push((column, row_offset * k + hot));
                }
            }
            visit(ring, &contributions);
        }
    }
    Ok(())
}

fn flush_wide<const D: usize>(
    wide: &mut [WideCyclotomicRing<<AkitaField as HasWide>::Wide, D>],
    reduced: &mut [CyclotomicRing<AkitaField, D>],
) {
    for (wide, reduced) in wide.iter_mut().zip(reduced) {
        *reduced += std::mem::replace(wide, WideCyclotomicRing::zero()).reduce();
    }
}

fn validate_block_geometry(
    segment_rings: usize,
    column_capacity: usize,
    num_positions: usize,
) -> Result<(usize, usize), AkitaError> {
    if num_positions == 0 || !num_positions.is_power_of_two() {
        return Err(AkitaError::InvalidInput(format!(
            "trace one-hot positions per block {num_positions} must be a nonzero power of two"
        )));
    }
    let total_rings = segment_rings
        .checked_mul(column_capacity)
        .ok_or_else(|| AkitaError::InvalidInput("trace one-hot ring count overflow".to_string()))?;
    Ok((total_rings, total_rings.div_ceil(num_positions)))
}

fn trace_block_task_parts<const D: usize>(
    one_hot_k: usize,
    num_positions: usize,
    blocks_per_column: usize,
) -> usize {
    let ring_alignment = (one_hot_k / D).max(1);
    debug_assert_eq!(num_positions % ring_alignment, 0);
    let max_parts = num_positions / ring_alignment;
    let target_tasks = rayon::current_num_threads()
        .saturating_mul(TASKS_PER_RAYON_WORKER)
        .max(1);
    target_tasks.div_ceil(blocks_per_column).clamp(1, max_parts)
}

fn trace_block_part_range(
    num_positions: usize,
    ring_alignment: usize,
    part: usize,
    parts: usize,
) -> (usize, usize) {
    debug_assert_eq!(num_positions % ring_alignment, 0);
    let aligned_positions = num_positions / ring_alignment;
    (
        part * aligned_positions / parts * ring_alignment,
        (part + 1) * aligned_positions / parts * ring_alignment,
    )
}

fn commit_packed<const D: usize>(
    backend: &CpuBackend,
    prepared: &<CpuBackend as ComputeBackendSetup<AkitaField>>::PreparedSetup,
    source: &TracePackedOneHot,
    plan: CommitInnerPlan,
) -> Result<CommitInnerWitness<AkitaField>, AkitaError> {
    let _span = tracing::info_span!(
        "TracePackedOneHot::commit_inner",
        ring_dimension = D,
        rows = source.rows.num_rows(),
        columns = source.rows.num_columns(),
        column_capacity = source.column_capacity,
    )
    .entered();
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
        let _accumulate_span = tracing::info_span!(
            "trace_onehot_commit_accumulate",
            blocks_per_column,
            task_parts = parts,
        )
        .entered();
        let partials = (0..blocks_per_column * parts)
            .into_par_iter()
            .map(|task| {
                let trace_block = task / parts;
                let part = task % parts;
                let mut wide = vec![WideCyclotomicRing::zero(); source.column_capacity * plan.n_a];
                let mut reduced = vec![CyclotomicRing::zero(); source.column_capacity * plan.n_a];
                let block_ring_start = trace_block * plan.num_positions_per_block;
                let (part_start, part_end) = trace_block_part_range(
                    plan.num_positions_per_block,
                    ring_alignment,
                    part,
                    parts,
                );
                let ring_start = block_ring_start + part_start;
                let ring_end = block_ring_start + part_end;
                let mut budget = 0usize;
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
                flush_wide(&mut wide, &mut reduced);
                Ok::<_, AkitaError>(reduced)
            })
            .collect::<Result<Vec<_>, _>>()?;
        drop(_accumulate_span);
        let mut rows = vec![vec![CyclotomicRing::zero(); plan.n_a]; num_blocks];
        for (task, block_rows) in partials.into_iter().enumerate() {
            let trace_block = task / parts;
            let part = task % parts;
            for column in 0..source.column_capacity {
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
        flush_wide(&mut wide, &mut reduced);
        reduced
            .chunks_exact(plan.n_a)
            .map(<[CyclotomicRing<AkitaField, D>]>::to_vec)
            .collect()
    };

    let _decompose_span = tracing::info_span!("trace_onehot_commit_decompose").entered();
    let digits = decompose_commit_blocks_into::<AkitaField, D>(
        &rows,
        plan.num_digits_outer,
        plan.log_basis_outer,
    )?;
    CommitInnerWitness::from_parts(rows, digits)
}

fn opening_fold_packed<const D: usize>(
    source: &TracePackedOneHot,
    plan: OpeningFoldPlan<'_, AkitaField, D>,
) -> Result<OpeningFoldOutput<AkitaField, D>, AkitaError> {
    let _span = tracing::info_span!(
        "TracePackedOneHot::evaluate_and_fold",
        ring_dimension = D,
        rows = source.rows.num_rows(),
        columns = source.rows.num_columns(),
        column_capacity = source.column_capacity,
    )
    .entered();
    let num_positions = match plan {
        OpeningFoldPlan::Base {
            num_positions_per_block,
            ..
        }
        | OpeningFoldPlan::Ring {
            num_positions_per_block,
            ..
        } => num_positions_per_block,
    };
    let segment_rings = source.segment_ring_elems::<D>()?;
    let (_, num_blocks) =
        validate_block_geometry(segment_rings, source.column_capacity, num_positions)?;
    let (live_weights, position_weights) = match plan {
        OpeningFoldPlan::Base {
            live_block_weights,
            position_weights,
            ..
        } => (live_block_weights.len(), position_weights.len()),
        OpeningFoldPlan::Ring {
            live_block_weights,
            position_weights,
            ..
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
                let mut folded = vec![CyclotomicRing::zero(); source.column_capacity];
                visit_segment_ring_range::<D>(
                    source,
                    ring_start,
                    ring_end,
                    |ring, contributions| {
                        let position = ring - block_ring_start;
                        match plan {
                            OpeningFoldPlan::Base {
                                position_weights, ..
                            } => {
                                let weight = position_weights[position];
                                for &(column, coefficient) in contributions {
                                    folded[column].coeffs[coefficient] += weight;
                                }
                            }
                            OpeningFoldPlan::Ring {
                                position_weights, ..
                            } => {
                                let weight = position_weights[position];
                                for &(column, coefficient) in contributions {
                                    weight.shift_accumulate_into(&mut folded[column], coefficient);
                                }
                            }
                        }
                    },
                )?;
                Ok::<_, AkitaError>(folded)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut folded = vec![CyclotomicRing::zero(); num_blocks];
        for (task, trace_folded) in partials.into_iter().enumerate() {
            let trace_block = task / parts;
            let part = task % parts;
            for column in 0..source.column_capacity {
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
        let mut folded = vec![CyclotomicRing::zero(); num_blocks];
        visit_segment_ring_range::<D>(source, 0, segment_rings, |ring, contributions| {
            for &(column, coefficient) in contributions {
                let global_ring = column * segment_rings + ring;
                let block = global_ring / num_positions;
                let position = global_ring % num_positions;
                match plan {
                    OpeningFoldPlan::Base {
                        position_weights, ..
                    } => {
                        folded[block].coeffs[coefficient] += position_weights[position];
                    }
                    OpeningFoldPlan::Ring {
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
    let eval = match plan {
        OpeningFoldPlan::Base {
            live_block_weights, ..
        } => folded
            .iter()
            .zip(live_block_weights)
            .fold(CyclotomicRing::zero(), |acc, (value, weight)| {
                acc + value.scale(weight)
            }),
        OpeningFoldPlan::Ring {
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

fn add_rotated_sparse<const D: usize>(
    dst: &mut [i32; D],
    challenge: &SparseChallenge,
    coefficient: usize,
) {
    for (&position, &value) in challenge.positions.iter().zip(&challenge.coeffs) {
        let shifted = position as usize + coefficient;
        if shifted < D {
            dst[shifted] += i32::from(value);
        } else {
            dst[shifted - D] -= i32::from(value);
        }
    }
}

fn decompose_fold_packed<const D: usize>(
    source: &TracePackedOneHot,
    challenges: &[SparseChallenge],
    num_positions: usize,
    num_digits: usize,
) -> Result<akita_prover::DecomposeFoldWitness<AkitaField>, AkitaError> {
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
    let compressed = if segment_rings >= num_positions {
        let blocks_per_column = segment_rings / num_positions;
        debug_assert_eq!(blocks_per_column * num_positions, segment_rings);
        let row_alignment = (source.one_hot_k / D).max(1);
        let target_tasks = rayon::current_num_threads().min(num_positions).max(1);
        let position_chunk = num_positions
            .div_ceil(target_tasks)
            .next_multiple_of(row_alignment);
        let position_starts = (0..num_positions)
            .step_by(position_chunk)
            .collect::<Vec<_>>();
        position_starts
            .into_par_iter()
            .map(|position_start| {
                let position_end = (position_start + position_chunk).min(num_positions);
                let mut compressed = vec![[0i32; D]; position_end - position_start];
                for trace_block in 0..blocks_per_column {
                    let ring_start = trace_block * num_positions + position_start;
                    let ring_end = trace_block * num_positions + position_end;
                    visit_segment_ring_range::<D>(
                        source,
                        ring_start,
                        ring_end,
                        |ring, contributions| {
                            let position = ring - trace_block * num_positions;
                            for &(column, coefficient) in contributions {
                                let block = column * blocks_per_column + trace_block;
                                add_rotated_sparse(
                                    &mut compressed[position - position_start],
                                    &challenges[block],
                                    coefficient,
                                );
                            }
                        },
                    )?;
                }
                Ok::<_, AkitaError>(compressed)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect()
    } else {
        let mut compressed = vec![[0i32; D]; num_positions];
        visit_segment_ring_range::<D>(source, 0, segment_rings, |ring, contributions| {
            for &(column, coefficient) in contributions {
                let global_ring = column * segment_rings + ring;
                let block = global_ring / num_positions;
                add_rotated_sparse(
                    &mut compressed[global_ring % num_positions],
                    &challenges[block],
                    coefficient,
                );
            }
        })?;
        compressed
    };
    let mut expanded = Vec::with_capacity(num_positions.saturating_mul(num_digits));
    for coeffs in compressed {
        expanded.push(coeffs);
        expanded.extend((1..num_digits).map(|_| [0i32; D]));
    }
    let modulus = (-AkitaField::one()).to_canonical_u128() + 1;
    Ok(build_decompose_fold_witness::<AkitaField, D>(
        expanded, modulus,
    ))
}

impl<const D: usize> RootCommitKernel<TracePackedOneHotView<'_, D>, AkitaField, D> for CpuBackend {
    fn commit_inner(
        &self,
        prepared: &Self::PreparedSetup,
        source: TracePackedOneHotView<'_, D>,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<AkitaField>, AkitaError> {
        commit_packed::<D>(self, prepared, source.source, plan)
    }
}

impl<const D: usize> OpeningFoldKernel<TracePackedOneHotView<'_, D>, AkitaField, D> for CpuBackend {
    fn evaluate_and_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: TracePackedOneHotView<'_, D>,
        plan: OpeningFoldPlan<'_, AkitaField, D>,
    ) -> Result<OpeningFoldOutput<AkitaField, D>, AkitaError> {
        opening_fold_packed(source.source, plan)
    }

    fn decompose_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: TracePackedOneHotView<'_, D>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<akita_prover::DecomposeFoldWitness<AkitaField>, AkitaError> {
        decompose_fold_packed::<D>(
            source.source,
            plan.challenges,
            plan.num_positions_per_block,
            plan.num_digits,
        )
    }
}

impl<const D: usize> OpeningBatchKernel<TracePackedOneHotBatchView<'_, D>, AkitaField, D>
    for CpuBackend
{
    fn decompose_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: TracePackedOneHotBatchView<'_, D>,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<BatchDecomposeFoldOutcome<AkitaField, D>, AkitaError> {
        let source = source.sources[0];
        match plan {
            DecomposeFoldBatchPlan::Sparse {
                challenges,
                num_positions_per_block,
                num_digits,
                ..
            } => Ok(BatchDecomposeFoldOutcome::Fused(
                decompose_fold_packed::<D>(
                    source,
                    challenges,
                    num_positions_per_block,
                    num_digits,
                )?,
            )),
            DecomposeFoldBatchPlan::Tensor { .. } => Ok(BatchDecomposeFoldOutcome::Unsupported),
        }
    }
}

impl<E, const D: usize> TensorProjectionKernel<TracePackedOneHotView<'_, D>, AkitaField, E, D>
    for CpuBackend
where
    E: ExtField<AkitaField>,
{
    fn column_partials(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        _source: TracePackedOneHotView<'_, D>,
        _logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<AkitaField>,
    {
        Err(AkitaError::UnsupportedSchedule(
            "Jolt trace one-hot sources require flat root challenges".to_string(),
        ))
    }

    fn packed_witness(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        _source: TracePackedOneHotView<'_, D>,
    ) -> Result<TensorPackedWitness<E>, AkitaError> {
        Err(AkitaError::UnsupportedSchedule(
            "Jolt trace one-hot sources require flat root challenges".to_string(),
        ))
    }

    fn root_projection(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        _source: TracePackedOneHotView<'_, D>,
    ) -> Result<RootTensorProjectionPoly<AkitaField>, AkitaError>
    where
        AkitaField: FromPrimitiveInt,
        E: FpExtEncoding<AkitaField>,
    {
        Err(AkitaError::UnsupportedSchedule(
            "Jolt trace one-hot sources require flat root challenges".to_string(),
        ))
    }
}

impl<E, const D: usize>
    TensorProjectionBatchKernel<TracePackedOneHotBatchView<'_, D>, AkitaField, E, D> for CpuBackend
where
    E: ExtField<AkitaField>,
{
    fn column_partials_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        _source: TracePackedOneHotBatchView<'_, D>,
        _logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<AkitaField>,
    {
        Err(AkitaError::UnsupportedSchedule(
            "Jolt trace one-hot sources require flat root challenges".to_string(),
        ))
    }

    fn sparse_linear_combination(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        _source: TracePackedOneHotBatchView<'_, D>,
        _coeffs: &[E],
    ) -> Result<
        Option<
            akita_prover::protocol::extension_opening_reduction::SparseExtensionOpeningWitness<E>,
        >,
        AkitaError,
    > {
        Err(AkitaError::UnsupportedSchedule(
            "Jolt trace one-hot sources require flat root challenges".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests assert valid kernel geometry")]

    use super::*;
    use akita_prover::OneHotPoly;

    #[derive(Debug)]
    struct TestRows {
        rows: usize,
        columns: usize,
        k: usize,
    }

    impl TraceOneHotRows for TestRows {
        fn num_rows(&self) -> usize {
            self.rows
        }

        fn num_columns(&self) -> usize {
            self.columns
        }

        fn fill_row(&self, row: usize, hot_lanes: &mut [u16]) {
            for (column, hot) in hot_lanes.iter_mut().enumerate() {
                *hot = ((row * (2 * column + 1) + column) % self.k) as u16;
            }
        }
    }

    fn assert_ring_mapping<const D: usize>(k: usize, rows: usize) {
        let source = TracePackedOneHot::new(
            k,
            64,
            8,
            Arc::new(TestRows {
                rows,
                columns: 3,
                k,
            }),
        )
        .unwrap();
        let segment_rings = source.segment_ring_elems::<D>().unwrap();
        let mut actual = Vec::new();
        visit_segment_ring_range::<D>(&source, 0, segment_rings, |ring, contributions| {
            actual.extend(
                contributions
                    .iter()
                    .map(|&(column, coefficient)| (column, ring * D + coefficient)),
            );
        })
        .unwrap();
        let expected = (0..rows)
            .flat_map(|row| {
                (0..3).map(move |column| (column, row * k + (row * (2 * column + 1) + column) % k))
            })
            .collect::<Vec<_>>();
        actual.sort_unstable();
        let mut expected = expected;
        expected.sort_unstable();
        assert_eq!(actual, expected);
    }

    #[test]
    fn row_major_mapping_is_dimension_generic() {
        for rows in [32, 64] {
            assert_ring_mapping::<64>(16, rows);
            assert_ring_mapping::<128>(16, rows);
            assert_ring_mapping::<256>(16, rows);
            assert_ring_mapping::<512>(16, rows);
            assert_ring_mapping::<64>(256, rows);
            assert_ring_mapping::<128>(256, rows);
            assert_ring_mapping::<256>(256, rows);
            assert_ring_mapping::<512>(256, rows);
        }
    }

    #[test]
    fn constructor_enforces_selector_capacity() {
        let rows = Arc::new(TestRows {
            rows: 32,
            columns: 9,
            k: 16,
        });
        assert!(TracePackedOneHot::new(16, 64, 8, rows).is_err());
    }

    fn assert_opening_kernels_match_materialized<const D: usize>(
        k: usize,
        rows: usize,
        num_positions: usize,
    ) {
        const COLUMNS: usize = 3;
        const CAPACITY: usize = 8;
        let source = TracePackedOneHot::new(
            k,
            64,
            CAPACITY,
            Arc::new(TestRows {
                rows,
                columns: COLUMNS,
                k,
            }),
        )
        .unwrap();
        let packed_indices = (0..CAPACITY)
            .flat_map(|column| {
                (0..rows).map(move |row| {
                    (column < COLUMNS).then_some(((row * (2 * column + 1) + column) % k) as u8)
                })
            })
            .collect();
        let materialized_source = OneHotPoly::<AkitaField, u8>::new(k, 64, packed_indices).unwrap();
        let num_blocks =
            <TracePackedOneHot as RootPolyShape<AkitaField, D>>::num_ring_elems(&source)
                / num_positions;
        let live_weights = (0..num_blocks)
            .map(|index| AkitaField::from_u64((index + 2) as u64))
            .collect::<Vec<_>>();
        let position_weights = (0..num_positions)
            .map(|index| AkitaField::from_u64((3 * index + 1) as u64))
            .collect::<Vec<_>>();
        let fold_plan = OpeningFoldPlan::Base {
            live_block_weights: &live_weights,
            position_weights: &position_weights,
            num_positions_per_block: num_positions,
        };
        let backend = CpuBackend;
        let streamed = <CpuBackend as OpeningFoldKernel<
            TracePackedOneHotView<'_, D>,
            AkitaField,
            D,
        >>::evaluate_and_fold(
            &backend,
            None,
            <TracePackedOneHot as RootOpeningSource<AkitaField, D>>::opening_view(&source).unwrap(),
            fold_plan,
        )
        .unwrap();
        let materialized = <CpuBackend as OpeningFoldKernel<_, AkitaField, D>>::evaluate_and_fold(
            &backend,
            None,
            <OneHotPoly<AkitaField, u8> as RootOpeningSource<AkitaField, D>>::opening_view(
                &materialized_source,
            )
            .unwrap(),
            fold_plan,
        )
        .unwrap();
        assert_eq!(streamed, materialized);

        let challenges = (0..num_blocks)
            .map(|block| SparseChallenge {
                positions: vec![0, (block % (D - 1) + 1) as u32],
                coeffs: vec![1, -1],
            })
            .collect::<Vec<_>>();
        let decompose_plan = DecomposeFoldPlan {
            challenges: &challenges,
            num_positions_per_block: num_positions,
            num_digits: 2,
            log_basis: 3,
        };
        let streamed = <CpuBackend as OpeningFoldKernel<
            TracePackedOneHotView<'_, D>,
            AkitaField,
            D,
        >>::decompose_fold(
            &backend,
            None,
            <TracePackedOneHot as RootOpeningSource<AkitaField, D>>::opening_view(&source).unwrap(),
            decompose_plan,
        )
        .unwrap();
        let materialized = <CpuBackend as OpeningFoldKernel<_, AkitaField, D>>::decompose_fold(
            &backend,
            None,
            <OneHotPoly<AkitaField, u8> as RootOpeningSource<AkitaField, D>>::opening_view(
                &materialized_source,
            )
            .unwrap(),
            decompose_plan,
        )
        .unwrap();
        assert_eq!(streamed, materialized);
    }

    #[test]
    fn blockwise_opening_kernels_match_materialized_onehot() {
        assert_opening_kernels_match_materialized::<64>(256, 32, 16);
        assert_opening_kernels_match_materialized::<512>(16, 32, 1);
    }
}
