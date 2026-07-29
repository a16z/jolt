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
use akita_prover::backend::poly_helpers::{build_decompose_fold_witness, fill_rotated_challenge};
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
const ROTATED_CHALLENGE_TABLE_BUDGET: usize = 1 << 26;
const DECOMPOSE_POSITION_WORKING_SET_TARGET: usize = 1 << 21;
const D64_K16_SHIFT_KEY_SPACE: usize = 1 << 16;
const SHARED_SHIFT_MIN_COLUMNS: u8 = 3;
const D64_K256_ROW_BATCH: usize = 1 << 13;
const D64_K256_ROW_FUSION: usize = 4;

type AkitaWideRing<const D: usize> = WideCyclotomicRing<<AkitaField as HasWide>::Wide, D>;

struct D64K16ShiftGroups {
    group_by_key: Vec<u8>,
    group_columns: Vec<u8>,
    group_counts: Vec<u8>,
    group_shifts: Vec<[usize; 4]>,
    touched_keys: Vec<u16>,
    partial_columns: Vec<u8>,
    num_groups: u8,
}

impl D64K16ShiftGroups {
    fn new(num_columns: usize) -> Option<Self> {
        if num_columns >= usize::from(u8::MAX) {
            return None;
        }
        Some(Self {
            group_by_key: vec![u8::MAX; D64_K16_SHIFT_KEY_SPACE],
            group_columns: vec![u8::MAX; num_columns * num_columns],
            group_counts: vec![0; num_columns],
            group_shifts: vec![[0; 4]; num_columns],
            touched_keys: Vec::with_capacity(num_columns),
            partial_columns: Vec::with_capacity(num_columns),
            num_groups: 0,
        })
    }

    fn build(&mut self, lanes: &[u16], num_columns: usize) -> bool {
        for key in self.touched_keys.drain(..) {
            self.group_by_key[usize::from(key)] = u8::MAX;
        }
        self.partial_columns.clear();
        self.num_groups = 0;
        if lanes.len() != 4 * num_columns {
            return false;
        }

        for column in 0..num_columns {
            let hot0 = lanes[column];
            let hot1 = lanes[num_columns + column];
            let hot2 = lanes[2 * num_columns + column];
            let hot3 = lanes[3 * num_columns + column];
            if [hot0, hot1, hot2, hot3].contains(&NO_HOT_LANE) {
                self.partial_columns.push(column as u8);
                continue;
            }
            let key = hot0 | (hot1 << 4) | (hot2 << 8) | (hot3 << 12);
            let mut group = self.group_by_key[usize::from(key)];
            if group == u8::MAX {
                group = self.num_groups;
                self.num_groups += 1;
                self.group_by_key[usize::from(key)] = group;
                self.touched_keys.push(key);
                self.group_counts[usize::from(group)] = 0;
                self.group_shifts[usize::from(group)] = [
                    usize::from(hot0),
                    16 + usize::from(hot1),
                    32 + usize::from(hot2),
                    48 + usize::from(hot3),
                ];
            }
            let group = usize::from(group);
            let count = usize::from(self.group_counts[group]);
            self.group_columns[group * num_columns + count] = column as u8;
            self.group_counts[group] += 1;
        }
        true
    }

    fn accumulate<const D: usize>(
        &self,
        src: &AkitaWideRing<D>,
        dst: &mut [AkitaWideRing<D>],
        a: usize,
        n_a: usize,
        lanes: &[u16],
        num_columns: usize,
    ) {
        for group in 0..self.num_groups {
            let group = usize::from(group);
            let count = usize::from(self.group_counts[group]);
            let columns = &self.group_columns[group * num_columns..group * num_columns + count];
            if self.group_counts[group] >= SHARED_SHIFT_MIN_COLUMNS {
                let shifts = &self.group_shifts[group];
                let src = src.coeffs();
                for coefficient in 0..D {
                    let shift = shifts[0];
                    let mut sum = if coefficient >= shift {
                        src[coefficient - shift]
                    } else {
                        -src[D + coefficient - shift]
                    };
                    for &shift in &shifts[1..] {
                        if coefficient >= shift {
                            sum += src[coefficient - shift];
                        } else {
                            sum -= src[D + coefficient - shift];
                        }
                    }
                    for &column in columns {
                        dst[usize::from(column) * n_a + a].coeffs_mut()[coefficient] += sum;
                    }
                }
            } else {
                for &column in columns {
                    src.shift_accumulate_array_into(
                        &mut dst[usize::from(column) * n_a + a],
                        &self.group_shifts[group],
                    );
                }
            }
        }
        for &column in &self.partial_columns {
            let column = usize::from(column);
            for (row_offset, row_lanes) in lanes.chunks_exact(num_columns).enumerate() {
                let hot = row_lanes[column];
                if hot != NO_HOT_LANE {
                    src.shift_accumulate_into(
                        &mut dst[column * n_a + a],
                        row_offset * 16 + usize::from(hot),
                    );
                }
            }
        }
    }
}

/// Row-major source for the semantic columns packed into `OneHotTrace`.
///
/// `fill_row` must overwrite all of `hot_lanes`. Use [`no_hot_lane`] when a
/// semantic column has no nonzero entry in that row.
pub trait TraceOneHotRows: Send + Sync + 'static {
    fn num_rows(&self) -> usize;
    fn num_columns(&self) -> usize;
    fn fill_row(&self, row: usize, hot_lanes: &mut [u16]);

    /// Bit `column` is set when that column has one hot lane in every row.
    fn always_present_mask(&self) -> u64 {
        0
    }

    /// Fills consecutive rows in row-major order, overwriting the entire buffer.
    fn fill_rows(&self, row_start: usize, hot_lanes: &mut [u16]) {
        let num_columns = self.num_columns();
        debug_assert_eq!(hot_lanes.len() % num_columns, 0);
        for (row_offset, row_lanes) in hot_lanes.chunks_exact_mut(num_columns).enumerate() {
            self.fill_row(row_start + row_offset, row_lanes);
        }
    }
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

/// Visits K<D ring elements as the raw lanes for the D/K trace rows packed
/// into each ring. This avoids expanding the row buffer into contribution
/// tuples when a kernel can consume lanes directly.
fn visit_segment_ring_lane_range<const D: usize>(
    source: &TracePackedOneHot,
    ring_start: usize,
    ring_end: usize,
    mut visit: impl FnMut(usize, &[u16]),
) -> Result<(), AkitaError> {
    validate_dimension::<D>(source.one_hot_k)?;
    if source.one_hot_k >= D {
        return Err(AkitaError::InvalidInput(format!(
            "trace one-hot raw-lane traversal requires K={} < D={D}",
            source.one_hot_k
        )));
    }
    let segment_rings = source.segment_ring_elems::<D>()?;
    if ring_start > ring_end || ring_end > segment_rings {
        return Err(AkitaError::InvalidInput(format!(
            "trace one-hot ring range {ring_start}..{ring_end} exceeds segment size {segment_rings}"
        )));
    }
    let rows_per_ring = D / source.one_hot_k;
    let num_columns = source.rows.num_columns();
    if !source.rows.num_rows().is_multiple_of(rows_per_ring) {
        return Err(AkitaError::InvalidInput(format!(
            "trace one-hot row count {} is not aligned to {rows_per_ring} rows per D={D} ring",
            source.rows.num_rows()
        )));
    }
    let lane_count = num_columns.checked_mul(rows_per_ring).ok_or_else(|| {
        AkitaError::InvalidInput("trace one-hot raw-lane buffer size overflow".to_string())
    })?;
    let mut lanes = vec![NO_HOT_LANE; lane_count];
    for ring in ring_start..ring_end {
        let row_start = ring * rows_per_ring;
        let populated_rows = source
            .rows
            .num_rows()
            .saturating_sub(row_start)
            .min(rows_per_ring);
        let populated_lanes = &mut lanes[..populated_rows * num_columns];
        source.rows.fill_rows(row_start, populated_lanes);
        for &hot in populated_lanes.iter() {
            if hot != NO_HOT_LANE && usize::from(hot) >= source.one_hot_k {
                return Err(AkitaError::InvalidInput(format!(
                    "trace one-hot lane {hot} is outside K={}",
                    source.one_hot_k
                )));
            }
        }
        visit(ring, populated_lanes);
    }
    Ok(())
}

fn flush_wide<const D: usize>(
    wide: &mut [AkitaWideRing<D>],
    reduced: &mut [CyclotomicRing<AkitaField, D>],
) {
    for (wide, reduced) in wide.iter_mut().zip(reduced) {
        *reduced += std::mem::replace(wide, WideCyclotomicRing::zero()).reduce();
    }
}

fn flush_wide_rank<const D: usize>(
    wide: &mut [AkitaWideRing<D>],
    reduced: &mut [CyclotomicRing<AkitaField, D>],
    n_a: usize,
    a: usize,
) {
    for (column, wide) in wide.iter_mut().enumerate() {
        let index = column * n_a + a;
        reduced[index] += std::mem::replace(wide, WideCyclotomicRing::zero()).reduce();
    }
}

#[inline(always)]
fn full_row_coefficients<const N: usize>(
    lanes: &[u16],
    num_columns: usize,
    column: usize,
    one_hot_k: usize,
) -> Option<[usize; N]> {
    if lanes.len() != N * num_columns {
        return None;
    }
    let coefficients =
        std::array::from_fn(|row| row * one_hot_k + usize::from(lanes[row * num_columns + column]));
    lanes
        .iter()
        .skip(column)
        .step_by(num_columns)
        .all(|&hot| hot != NO_HOT_LANE)
        .then_some(coefficients)
}

#[inline(always)]
fn shift_accumulate_full_rows<const D: usize, const N: usize>(
    src: &AkitaWideRing<D>,
    dst: &mut AkitaWideRing<D>,
    lanes: &[u16],
    num_columns: usize,
    column: usize,
    one_hot_k: usize,
) -> bool {
    let Some(coefficients) = full_row_coefficients::<N>(lanes, num_columns, column, one_hot_k)
    else {
        return false;
    };
    src.shift_accumulate_array_into(dst, &coefficients);
    true
}

#[inline(always)]
fn try_shift_accumulate_full_rows<const D: usize>(
    src: &AkitaWideRing<D>,
    dst: &mut AkitaWideRing<D>,
    lanes: &[u16],
    num_columns: usize,
    column: usize,
    one_hot_k: usize,
    rows_per_ring: usize,
) -> bool {
    match rows_per_ring {
        2 => shift_accumulate_full_rows::<D, 2>(src, dst, lanes, num_columns, column, one_hot_k),
        4 => shift_accumulate_full_rows::<D, 4>(src, dst, lanes, num_columns, column, one_hot_k),
        8 => shift_accumulate_full_rows::<D, 8>(src, dst, lanes, num_columns, column, one_hot_k),
        16 => shift_accumulate_full_rows::<D, 16>(src, dst, lanes, num_columns, column, one_hot_k),
        32 => shift_accumulate_full_rows::<D, 32>(src, dst, lanes, num_columns, column, one_hot_k),
        _ => false,
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

#[inline(always)]
fn shift_accumulate_distinct_sources<const D: usize, const N: usize>(
    sources: &[AkitaWideRing<D>],
    source_indices: &[usize; N],
    shifts: &[usize; N],
    dst: &mut AkitaWideRing<D>,
) {
    debug_assert!(N >= 2);
    for (coefficient, dst) in dst.coeffs_mut().iter_mut().enumerate() {
        // Two independent chains hide vector-add latency before the final merge.
        let shift0 = shifts[0];
        let source0 = sources[source_indices[0]].coeffs();
        let mut sum0 = if coefficient >= shift0 {
            source0[coefficient - shift0]
        } else {
            -source0[D + coefficient - shift0]
        };
        let shift1 = shifts[1];
        let source1 = sources[source_indices[1]].coeffs();
        let mut sum1 = if coefficient >= shift1 {
            source1[coefficient - shift1]
        } else {
            -source1[D + coefficient - shift1]
        };
        for entry in 2..N {
            let shift = shifts[entry];
            let source = sources[source_indices[entry]].coeffs();
            if coefficient >= shift {
                if entry.is_multiple_of(2) {
                    sum0 += source[coefficient - shift];
                } else {
                    sum1 += source[coefficient - shift];
                }
            } else if entry.is_multiple_of(2) {
                sum0 -= source[D + coefficient - shift];
            } else {
                sum1 -= source[D + coefficient - shift];
            }
        }
        sum0 += sum1;
        *dst += sum0;
    }
}

#[inline(always)]
fn shift_accumulate_source_batch<const D: usize>(
    sources: &[AkitaWideRing<D>],
    source_indices: &[usize; D64_K256_ROW_FUSION],
    shifts: &[usize; D64_K256_ROW_FUSION],
    count: usize,
    dst: &mut AkitaWideRing<D>,
) {
    macro_rules! accumulate {
        ($count:literal) => {
            shift_accumulate_distinct_sources::<D, $count>(
                sources,
                &std::array::from_fn(|index| source_indices[index]),
                &std::array::from_fn(|index| shifts[index]),
                dst,
            )
        };
    }
    match count {
        0 => {}
        1 => sources[source_indices[0]].shift_accumulate_into(dst, shifts[0]),
        2 => accumulate!(2),
        3 => accumulate!(3),
        4 => accumulate!(4),
        _ => unreachable!("K=256 row fusion emits at most four contributions"),
    }
}

/// Batches four consecutive trace rows for columns that are known to be present in every row.
/// This is isolated from the dimension-generic fallback because its source tile assumes K/D = 4.
fn commit_d64_k256_hybrid<const D: usize>(
    source: &TracePackedOneHot,
    a_rows: &[&[CyclotomicRing<AkitaField, D>]],
    plan: CommitInnerPlan,
    segment_rings: usize,
    num_blocks: usize,
) -> Result<Vec<Vec<CyclotomicRing<AkitaField, D>>>, AkitaError> {
    debug_assert_eq!(D, 64);
    debug_assert_eq!(source.one_hot_k, 256);
    let rings_per_row = source.one_hot_k / D;
    debug_assert_eq!(rings_per_row, 4);
    let blocks_per_column = segment_rings / plan.num_positions_per_block;
    let rows_per_block = plan.num_positions_per_block / rings_per_row;
    let num_columns = source.rows.num_columns();
    let block_batch = 1;
    let row_batch = D64_K256_ROW_BATCH;
    let row_fusion = D64_K256_ROW_FUSION;
    let active_columns_mask = if num_columns == u32::BITS as usize {
        u32::MAX
    } else {
        (1u32 << num_columns) - 1
    };
    let always_present_mask = source.rows.always_present_mask();
    if always_present_mask >> num_columns != 0 {
        return Err(AkitaError::InvalidInput(format!(
            "trace always-present mask has columns outside the declared {num_columns} columns"
        )));
    }
    let dense_mask = always_present_mask as u32;
    let sparse_mask = active_columns_mask & !dense_mask;
    let block_groups = blocks_per_column.div_ceil(block_batch);
    let parts =
        trace_block_task_parts::<D>(source.one_hot_k, plan.num_positions_per_block, block_groups);
    let _accumulate_span = tracing::info_span!(
        "trace_onehot_commit_accumulate",
        num_blocks,
        blocks_per_column,
        block_batch,
        row_batch,
        row_fusion,
        block_groups,
        task_parts = parts,
        tasks = block_groups * parts,
        active_columns = num_columns,
        dense_columns = dense_mask.count_ones(),
        sparse_columns = sparse_mask.count_ones(),
        rows_per_ring = D / source.one_hot_k,
        fused_dense_rows = true,
    )
    .entered();
    let partials = (0..block_groups * parts)
        .into_par_iter()
        .map(|task| {
            let block_group = task / parts;
            let part = task % parts;
            let block_start = block_group * block_batch;
            let group_len = (blocks_per_column - block_start).min(block_batch);
            let (part_start, part_end) =
                trace_block_part_range(plan.num_positions_per_block, rings_per_row, part, parts);
            let row_start = part_start / rings_per_row;
            let row_end = part_end / rings_per_row;
            let mut reduced = vec![CyclotomicRing::zero(); group_len * num_columns * plan.n_a];
            let mut rank_wide = vec![AkitaWideRing::<D>::zero(); group_len * num_columns];
            let mut lanes = vec![NO_HOT_LANE; num_columns];
            let mut hot_values = vec![0u8; group_len * row_batch * num_columns];
            let mut present_masks = vec![0u32; group_len * row_batch];
            let mut source_tile = vec![AkitaWideRing::<D>::zero(); row_fusion * rings_per_row];
            let mut source_indices = [0usize; D64_K256_ROW_FUSION];
            let mut shifts = [0usize; D64_K256_ROW_FUSION];

            for tile_start in (row_start..row_end).step_by(row_batch) {
                let tile_len = (row_end - tile_start).min(row_batch);
                for block_offset in 0..group_len {
                    let trace_block = block_start + block_offset;
                    let trace_row_start = trace_block * rows_per_block + tile_start;
                    for row_offset in 0..tile_len {
                        source
                            .rows
                            .fill_row(trace_row_start + row_offset, &mut lanes);
                        let staged_row = block_offset * row_batch + row_offset;
                        present_masks[staged_row] = 0;
                        for (column, &hot) in lanes.iter().enumerate() {
                            if hot == NO_HOT_LANE {
                                if dense_mask & (1 << column) != 0 {
                                    return Err(AkitaError::InvalidInput(format!(
                                        "trace column {column} was declared always present but row {} has no hot lane",
                                        trace_row_start + row_offset
                                    )));
                                }
                                continue;
                            }
                            if usize::from(hot) >= source.one_hot_k {
                                return Err(AkitaError::InvalidInput(format!(
                                    "trace one-hot lane {hot} is outside K={}",
                                    source.one_hot_k
                                )));
                            }
                            hot_values[staged_row * num_columns + column] = hot as u8;
                            present_masks[staged_row] |= 1 << column;
                        }
                    }
                }

                for (a, a_row) in a_rows.iter().enumerate() {
                    for fusion_start in (0..tile_len).step_by(row_fusion) {
                        let fusion_len = (tile_len - fusion_start).min(row_fusion);
                        for row_offset in 0..fusion_len {
                            let tile_row = fusion_start + row_offset;
                            for quarter in 0..rings_per_row {
                                let position = (tile_start + tile_row) * rings_per_row + quarter;
                                let a_col = position * plan.num_digits_inner;
                                source_tile[row_offset * rings_per_row + quarter] =
                                    WideCyclotomicRing::from_ring(&a_row[a_col]);
                            }
                        }
                        for block_offset in 0..group_len {
                            let mut remaining_dense = dense_mask;
                            while remaining_dense != 0 {
                                let column = remaining_dense.trailing_zeros() as usize;
                                remaining_dense &= remaining_dense - 1;
                                for row_offset in 0..fusion_len {
                                    let staged_row =
                                        block_offset * row_batch + fusion_start + row_offset;
                                    let hot =
                                        hot_values[staged_row * num_columns + column] as usize;
                                    source_indices[row_offset] =
                                        row_offset * rings_per_row + hot / D;
                                    shifts[row_offset] = hot % D;
                                }
                                shift_accumulate_source_batch(
                                    &source_tile,
                                    &source_indices,
                                    &shifts,
                                    fusion_len,
                                    &mut rank_wide[block_offset * num_columns + column],
                                );
                            }
                            for row_offset in 0..fusion_len {
                                let staged_row =
                                    block_offset * row_batch + fusion_start + row_offset;
                                let mut remaining = present_masks[staged_row] & sparse_mask;
                                while remaining != 0 {
                                    let column = remaining.trailing_zeros() as usize;
                                    remaining &= remaining - 1;
                                    let hot =
                                        hot_values[staged_row * num_columns + column] as usize;
                                    source_tile[row_offset * rings_per_row + hot / D]
                                        .shift_accumulate_into(
                                            &mut rank_wide[block_offset * num_columns + column],
                                            hot % D,
                                        );
                                }
                            }
                        }
                    }
                    flush_wide_rank(&mut rank_wide, &mut reduced, plan.n_a, a);
                }
            }
            Ok::<_, AkitaError>(reduced)
        })
        .collect::<Result<Vec<_>, _>>()?;
    drop(_accumulate_span);

    let _merge_span = tracing::info_span!(
        "trace_onehot_commit_merge_partials",
        num_blocks,
        blocks_per_column,
        block_batch,
        block_groups,
        task_parts = parts,
        active_columns = num_columns,
        n_a = plan.n_a,
    )
    .entered();
    let mut rows = vec![vec![CyclotomicRing::zero(); plan.n_a]; num_blocks];
    for (task, block_rows) in partials.into_iter().enumerate() {
        let block_group = task / parts;
        let part = task % parts;
        let block_start = block_group * block_batch;
        let group_len = (blocks_per_column - block_start).min(block_batch);
        for block_offset in 0..group_len {
            let trace_block = block_start + block_offset;
            for column in 0..num_columns {
                let dst = &mut rows[column * blocks_per_column + trace_block];
                let src_index = (block_offset * num_columns + column) * plan.n_a;
                let src = &block_rows[src_index..src_index + plan.n_a];
                if part == 0 {
                    dst.copy_from_slice(src);
                } else {
                    for (dst, src) in dst.iter_mut().zip(src) {
                        *dst += *src;
                    }
                }
            }
        }
    }
    Ok(rows)
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
        one_hot_k = source.one_hot_k,
        rows = source.rows.num_rows(),
        columns = source.rows.num_columns(),
        column_capacity = source.column_capacity,
        n_a = plan.n_a,
        positions_per_block = plan.num_positions_per_block,
        inner_digits = plan.num_digits_inner,
        outer_digits = plan.num_digits_outer,
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
    let a_matrix = expanded.shared_matrix().covering_at_dyn(
        plan.n_a
            .checked_mul(active_cols)
            .ok_or_else(|| AkitaError::InvalidSetup("active A extent overflow".to_string()))?,
        D,
    )?;
    let a_view = a_matrix.ring_view::<D>(plan.n_a, active_cols)?;
    let a_rows = a_view.rows().collect::<Vec<_>>();
    let max_per_ring = (D / source.one_hot_k).max(1);
    drop(_prepare_span);

    let rows = if segment_rings >= plan.num_positions_per_block
        && D == 64
        && source.one_hot_k == 256
        && source.rows.num_columns() <= u32::BITS as usize
        && source.rows.always_present_mask() != 0
    {
        commit_d64_k256_hybrid(source, &a_rows, plan, segment_rings, num_blocks)?
    } else if segment_rings >= plan.num_positions_per_block {
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
            fused_four_shifts = D == 64 && source.one_hot_k == 16,
            shared_shift_groups = D == 64 && source.one_hot_k == 16,
            generic_fused_row_shifts =
                D != 64 && matches!(D / source.one_hot_k, 2 | 4 | 8 | 16 | 32),
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
                let rank_tiled_k256 =
                    D == 64 && source.one_hot_k == 256 && num_columns <= u32::BITS as usize;
                let mut wide = if rank_tiled_k256 {
                    Vec::new()
                } else {
                    vec![WideCyclotomicRing::zero(); num_columns * plan.n_a]
                };
                let mut budget = 0usize;
                if source.one_hot_k < D {
                    let rows_per_ring = D / source.one_hot_k;
                    let fuse_four_shifts = D == 64 && source.one_hot_k == 16;
                    let mut shift_groups = fuse_four_shifts
                        .then(|| D64K16ShiftGroups::new(num_columns))
                        .flatten();
                    visit_segment_ring_lane_range::<D>(
                        source,
                        ring_start,
                        ring_end,
                        |ring, lanes| {
                            let position = ring - block_ring_start;
                            let a_col = position * plan.num_digits_inner;
                            let grouped = shift_groups
                                .as_mut()
                                .is_some_and(|groups| groups.build(lanes, num_columns));
                            for (a, a_row) in a_rows.iter().enumerate() {
                                let a_wide = WideCyclotomicRing::from_ring(&a_row[a_col]);
                                if grouped {
                                    if let Some(groups) = &shift_groups {
                                        groups.accumulate(
                                            &a_wide,
                                            &mut wide,
                                            a,
                                            plan.n_a,
                                            lanes,
                                            num_columns,
                                        );
                                        continue;
                                    }
                                }
                                if fuse_four_shifts {
                                    debug_assert_eq!(rows_per_ring, 4);
                                    for column in 0..num_columns {
                                        let hot0 = lanes[column];
                                        let hot1 = lanes[num_columns + column];
                                        let hot2 = lanes[2 * num_columns + column];
                                        let hot3 = lanes[3 * num_columns + column];
                                        if hot0 != NO_HOT_LANE
                                            && hot1 != NO_HOT_LANE
                                            && hot2 != NO_HOT_LANE
                                            && hot3 != NO_HOT_LANE
                                        {
                                            let shifts = [
                                                usize::from(hot0),
                                                16 + usize::from(hot1),
                                                32 + usize::from(hot2),
                                                48 + usize::from(hot3),
                                            ];
                                            a_wide.shift_accumulate_array_into(
                                                &mut wide[column * plan.n_a + a],
                                                &shifts,
                                            );
                                        } else {
                                            for (row_offset, hot) in
                                                [hot0, hot1, hot2, hot3].into_iter().enumerate()
                                            {
                                                if hot == NO_HOT_LANE {
                                                    continue;
                                                }
                                                a_wide.shift_accumulate_into(
                                                    &mut wide[column * plan.n_a + a],
                                                    row_offset * 16 + usize::from(hot),
                                                );
                                            }
                                        }
                                    }
                                } else {
                                    for column in 0..num_columns {
                                        let dst = &mut wide[column * plan.n_a + a];
                                        if try_shift_accumulate_full_rows(
                                            &a_wide,
                                            dst,
                                            lanes,
                                            num_columns,
                                            column,
                                            source.one_hot_k,
                                            rows_per_ring,
                                        ) {
                                            continue;
                                        }
                                        for (row_offset, row_lanes) in
                                            lanes.chunks_exact(num_columns).enumerate()
                                        {
                                            let hot = row_lanes[column];
                                            if hot != NO_HOT_LANE {
                                                a_wide.shift_accumulate_into(
                                                    dst,
                                                    row_offset * source.one_hot_k
                                                        + usize::from(hot),
                                                );
                                            }
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
                    // Stream one A rank at a time so the active wide accumulators stay in cache.
                    let rings_per_row = source.one_hot_k / D;
                    debug_assert_eq!(rings_per_row, 4);
                    debug_assert_eq!(ring_start % rings_per_row, 0);
                    debug_assert_eq!(ring_end % rings_per_row, 0);
                    let row_start = ring_start / rings_per_row;
                    let row_end = ring_end / rings_per_row;
                    let mut lanes = vec![NO_HOT_LANE; num_columns];
                    let mut hot_values = vec![0u8; D64_K256_ROW_BATCH * num_columns];
                    let mut quarter_masks = vec![[0u32; 4]; D64_K256_ROW_BATCH];
                    let mut rank_wide = vec![WideCyclotomicRing::zero(); num_columns];
                    for tile_start in (row_start..row_end).step_by(D64_K256_ROW_BATCH) {
                        let tile_len = (row_end - tile_start).min(D64_K256_ROW_BATCH);
                        for row_offset in 0..tile_len {
                            source.rows.fill_row(tile_start + row_offset, &mut lanes);
                            let masks = &mut quarter_masks[row_offset];
                            *masks = [0; 4];
                            for (column, &hot) in lanes.iter().enumerate() {
                                if hot == NO_HOT_LANE {
                                    continue;
                                }
                                if usize::from(hot) >= source.one_hot_k {
                                    return Err(AkitaError::InvalidInput(format!(
                                        "trace one-hot lane {hot} is outside K={}",
                                        source.one_hot_k
                                    )));
                                }
                                hot_values[row_offset * num_columns + column] = hot as u8;
                                masks[usize::from(hot) / D] |= 1 << column;
                            }
                        }
                        for (a, a_row) in a_rows.iter().enumerate() {
                            for row_offset in 0..tile_len {
                                let trace_row = tile_start + row_offset;
                                for (quarter, &mask) in quarter_masks[row_offset].iter().enumerate()
                                {
                                    if mask == 0 {
                                        continue;
                                    }
                                    let ring = trace_row * rings_per_row + quarter;
                                    let position = ring - block_ring_start;
                                    let a_col = position * plan.num_digits_inner;
                                    let a_wide = WideCyclotomicRing::from_ring(&a_row[a_col]);
                                    let mut remaining = mask;
                                    while remaining != 0 {
                                        let column = remaining.trailing_zeros() as usize;
                                        remaining &= remaining - 1;
                                        let hot =
                                            hot_values[row_offset * num_columns + column] as usize;
                                        a_wide
                                            .shift_accumulate_into(&mut rank_wide[column], hot % D);
                                    }
                                }
                            }
                            flush_wide_rank(&mut rank_wide, &mut reduced, plan.n_a, a);
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

    let _decompose_span = tracing::info_span!(
        "trace_onehot_commit_decompose",
        num_blocks,
        n_a = plan.n_a,
        outer_digits = plan.num_digits_outer,
        outer_log_basis = plan.log_basis_outer,
    )
    .entered();
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
    let weight_kind = match plan {
        OpeningFoldPlan::Base { .. } => "base",
        OpeningFoldPlan::Ring { .. } => "ring",
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
                    visit_segment_ring_lane_range::<D>(
                        source,
                        ring_start,
                        ring_end,
                        |ring, lanes| {
                            let position = ring - block_ring_start;
                            match plan {
                                OpeningFoldPlan::Base {
                                    position_weights, ..
                                } => {
                                    let weight = position_weights[position];
                                    for (row_offset, row_lanes) in
                                        lanes.chunks_exact(num_columns).enumerate()
                                    {
                                        let coefficient_base = row_offset * source.one_hot_k;
                                        for (column, &hot) in row_lanes.iter().enumerate() {
                                            if hot != NO_HOT_LANE {
                                                folded[column].coeffs
                                                    [coefficient_base + usize::from(hot)] += weight;
                                            }
                                        }
                                    }
                                }
                                OpeningFoldPlan::Ring {
                                    position_weights, ..
                                } => {
                                    let weight = position_weights[position];
                                    for (row_offset, row_lanes) in
                                        lanes.chunks_exact(num_columns).enumerate()
                                    {
                                        let coefficient_base = row_offset * source.one_hot_k;
                                        for (column, &hot) in row_lanes.iter().enumerate() {
                                            if hot != NO_HOT_LANE {
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
    let _reduce_span = tracing::info_span!(
        "trace_onehot_evaluate_fold_reduce_blocks",
        num_blocks,
        weight_kind,
    )
    .entered();
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecomposeRotationMode {
    Auto,
    Dense,
    Sparse,
}

impl DecomposeRotationMode {
    fn from_env() -> Result<Self, AkitaError> {
        match std::env::var("JOLT_AKITA_DECOMPOSE_MODE").as_deref() {
            Ok("dense") => Ok(Self::Dense),
            Ok("sparse") => Ok(Self::Sparse),
            Ok("auto") | Err(std::env::VarError::NotPresent) => Ok(Self::Auto),
            Ok(value) => Err(AkitaError::InvalidInput(format!(
                "JOLT_AKITA_DECOMPOSE_MODE must be auto, dense, or sparse; got {value:?}"
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

struct PreparedSparseChallenge {
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

enum PreparedRotations<const D: usize> {
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

fn prepare_rotations<const D: usize>(
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
    let use_dense = match mode {
        DecomposeRotationMode::Auto => D == 64 && dense_bytes <= ROTATED_CHALLENGE_TABLE_BUDGET,
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
fn add_rotated_contributions<const D: usize>(
    dst: &mut [i32; D],
    contributions: &[(usize, usize)],
    rotations: &PreparedRotations<D>,
    trace_block: usize,
    num_columns: usize,
) {
    match rotations {
        PreparedRotations::Dense(rotated) => {
            let table = |&(column, coefficient): &(usize, usize)| {
                &rotated[((trace_block * num_columns + column) * D) + coefficient]
            };
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
                [entry0, entry1, entry2, entry3, entry4, entry5] => {
                    add_rotated_dense_tables(
                        dst,
                        [
                            table(entry0),
                            table(entry1),
                            table(entry2),
                            table(entry3),
                            table(entry4),
                            table(entry5),
                        ],
                    );
                }
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

fn decompose_fold_packed_with_mode<const D: usize>(
    source: &TracePackedOneHot,
    challenges: &[SparseChallenge],
    num_positions: usize,
    num_digits: usize,
    rotation_mode: DecomposeRotationMode,
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
        let _compress_span = tracing::info_span!(
            "trace_onehot_decompose_accumulate",
            mode = "position_parallel",
            num_blocks,
            blocks_per_column,
            position_tasks,
            position_chunk,
            position_working_set_bytes = position_chunk * std::mem::size_of::<[i32; D]>(),
            dense_rotations = rotations.is_dense(),
        )
        .entered();
        let mut compressed = vec![[0i32; D]; num_positions];
        compressed
            .par_chunks_mut(position_chunk)
            .enumerate()
            .try_for_each(|(position_task, compressed)| {
                let position_start = position_task * position_chunk;
                let position_end = position_start + compressed.len();
                for trace_block in 0..blocks_per_column {
                    let ring_start = trace_block * num_positions + position_start;
                    let ring_end = trace_block * num_positions + position_end;
                    if source.one_hot_k < D {
                        let num_columns = source.rows.num_columns();
                        let rows_per_ring = D / source.one_hot_k;
                        let mut coefficients = Vec::with_capacity(rows_per_ring);
                        visit_segment_ring_lane_range::<D>(
                            source,
                            ring_start,
                            ring_end,
                            |ring, lanes| {
                                let position = ring - trace_block * num_positions;
                                let dst = &mut compressed[position - position_start];
                                if rows_per_ring <= 4 {
                                    for column in 0..num_columns {
                                        let mut fixed_coefficients = [0usize; 4];
                                        let mut count = 0;
                                        for (row_offset, row_lanes) in
                                            lanes.chunks_exact(num_columns).enumerate()
                                        {
                                            let hot = row_lanes[column];
                                            if hot != NO_HOT_LANE {
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
                                        for (row_offset, row_lanes) in
                                            lanes.chunks_exact(num_columns).enumerate()
                                        {
                                            let hot = row_lanes[column];
                                            if hot != NO_HOT_LANE {
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

fn decompose_fold_packed<const D: usize>(
    source: &TracePackedOneHot,
    challenges: &[SparseChallenge],
    num_positions: usize,
    num_digits: usize,
) -> Result<akita_prover::DecomposeFoldWitness<AkitaField>, AkitaError> {
    decompose_fold_packed_with_mode::<D>(
        source,
        challenges,
        num_positions,
        num_digits,
        DecomposeRotationMode::from_env()?,
    )
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
        let dense = decompose_fold_packed_with_mode::<D>(
            &source,
            &challenges,
            num_positions,
            2,
            DecomposeRotationMode::Dense,
        )
        .unwrap();
        let sparse = decompose_fold_packed_with_mode::<D>(
            &source,
            &challenges,
            num_positions,
            2,
            DecomposeRotationMode::Sparse,
        )
        .unwrap();
        assert_eq!(dense, materialized);
        assert_eq!(sparse, materialized);
    }

    #[test]
    fn blockwise_opening_kernels_match_materialized_onehot() {
        assert_opening_kernels_match_materialized::<64>(256, 32, 16);
        assert_opening_kernels_match_materialized::<128>(256, 32, 16);
        assert_opening_kernels_match_materialized::<256>(256, 32, 16);
        assert_opening_kernels_match_materialized::<512>(256, 32, 8);
        assert_opening_kernels_match_materialized::<64>(16, 32, 4);
        assert_opening_kernels_match_materialized::<128>(16, 32, 2);
        assert_opening_kernels_match_materialized::<256>(16, 32, 2);
        assert_opening_kernels_match_materialized::<512>(16, 32, 1);
        assert_opening_kernels_match_materialized::<64>(16, 32, 16);
        assert_opening_kernels_match_materialized::<128>(16, 32, 8);
        assert_opening_kernels_match_materialized::<256>(16, 32, 4);
        assert_opening_kernels_match_materialized::<512>(16, 32, 2);
    }
}
