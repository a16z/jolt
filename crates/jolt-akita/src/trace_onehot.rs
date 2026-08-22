//! Streaming kernels for Jolt's prefix-packed trace one-hot polynomial.
//!
//! A trace row gives the selected one-hot row for each column. Byte zero normally
//! denotes no stored coefficient; a per-row column mask distinguishes selected row
//! zero for families such as RAM. Jolt's kernels consume that row-major source
//! directly and lay the columns out as consecutive `K * T` segments inside one
//! physical polynomial. Padding selector slots are zero.

#![expect(
    clippy::indexing_slicing,
    reason = "hot kernels index geometry validated by TracePackedOneHot and their plans"
)]

use std::fmt;
use std::ops::Deref;
use std::sync::{Arc, RwLock, RwLockReadGuard};

use akita_algebra::ring::WideCyclotomicRing;
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_field::unreduced::HasWide;
use akita_field::{
    AkitaError, CanonicalField, ExtField, FromPrimitiveInt, MulBaseUnreduced, PseudoMersenneField,
};
use akita_prover::backend::poly_helpers::{build_decompose_fold_witness, fill_rotated_challenge};
use akita_prover::backend::{DenseBatchView, DenseView, OneHotBatchView, OneHotView};
use akita_prover::compute::{
    CommitInnerPlan, DecomposeFoldBatchPlan, DecomposeFoldPlan, OpeningBatchKernel,
    OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan, RootCommitKernel, TensorPackedWitness,
    TensorProjectionBatchKernel, TensorProjectionKernel,
};
use akita_prover::{
    BatchDecomposeFoldOutcome, CommitInnerWitness, ComputeBackendSetup, CpuBackend, DensePoly,
    OneHotPoly, RootCommitSource, RootOpeningSource, RootPolyMeta, RootPolyShape,
    RootTensorProjectionPoly, RootTensorSource,
};
use akita_types::FpExtEncoding;
use rayon::prelude::*;

use crate::AkitaField;

const NO_SELECTED_ROW: u8 = 0;
const MAX_WIDE_ACCUMULATIONS: usize = 1 << 15;
const TASKS_PER_RAYON_WORKER: usize = 4;
const ROTATED_CHALLENGE_TABLE_BUDGET: usize = 1 << 28;
const DECOMPOSE_POSITION_WORKING_SET_TARGET: usize = 1 << 21;
const SHARED_SHIFT_MIN_COLUMNS: u8 = 3;
const K256_ROW_BATCH: usize = 1 << 13;
const _: () = assert!(K256_ROW_BATCH <= i16::MAX as usize);

#[inline(always)]
fn row_is_committed(selected_row: u8, committed_zero_mask: u64, column: usize) -> bool {
    selected_row != NO_SELECTED_ROW || committed_zero_mask & (1u64 << column) != 0
}

type AkitaWideRing<const D: usize> = WideCyclotomicRing<<AkitaField as HasWide>::Wide, D>;

// Canonical reduction on every add costs more than tracking 2^128 wraps and
// applying 2^128 = MODULUS_OFFSET only when the tile is flushed.
#[derive(Clone)]
struct DeferredFp128Ring<const D: usize> {
    lo: [u64; D],
    hi: [u64; D],
    wraps: [i16; D],
}

impl<const D: usize> DeferredFp128Ring<D> {
    fn zero() -> Self {
        Self {
            lo: [0; D],
            hi: [0; D],
            wraps: [0; D],
        }
    }

    #[inline(always)]
    fn add_coefficient(&mut self, index: usize, value: AkitaField) {
        let [value_lo, value_hi] = value.to_limbs();
        let (lo, carry_lo) = self.lo[index].overflowing_add(value_lo);
        let (hi, carry_hi) = self.hi[index].carrying_add(value_hi, carry_lo);
        self.lo[index] = lo;
        self.hi[index] = hi;
        self.wraps[index] += i16::from(carry_hi);
    }

    #[inline(always)]
    fn sub_coefficient(&mut self, index: usize, value: AkitaField) {
        let [value_lo, value_hi] = value.to_limbs();
        let (lo, borrow_lo) = self.lo[index].overflowing_sub(value_lo);
        let (hi, borrow_hi) = self.hi[index].borrowing_sub(value_hi, borrow_lo);
        self.lo[index] = lo;
        self.hi[index] = hi;
        self.wraps[index] -= i16::from(borrow_hi);
    }

    #[inline(always)]
    fn shift_accumulate(&mut self, source: &CyclotomicRing<AkitaField, D>, shift: usize) {
        debug_assert!(shift < D);
        let (lo, hi) = source.coefficients().split_at(D - shift);
        for (index, &value) in lo.iter().enumerate() {
            self.add_coefficient(index + shift, value);
        }
        for (index, &value) in hi.iter().enumerate() {
            self.sub_coefficient(index, value);
        }
    }

    fn reduce_and_clear(&mut self) -> CyclotomicRing<AkitaField, D> {
        CyclotomicRing::from_coefficients(std::array::from_fn(|index| {
            let lo = std::mem::take(&mut self.lo[index]);
            let hi = std::mem::take(&mut self.hi[index]);
            let wraps = std::mem::take(&mut self.wraps[index]);
            debug_assert!(usize::from(wraps.unsigned_abs()) <= K256_ROW_BATCH);

            let base =
                AkitaField::from_canonical_u128_reduced(u128::from(lo) | (u128::from(hi) << 64));
            let correction = AkitaField::from_canonical_u128_reduced(
                u128::from(wraps.unsigned_abs()) * AkitaField::MODULUS_OFFSET,
            );
            if wraps >= 0 {
                base + correction
            } else {
                base - correction
            }
        }))
    }
}

/// Groups columns that share four consecutive K=16 row shifts. Adaptive
/// dimensions use one, two, or four of these groups per ring, preserving the
/// useful four-row reuse pattern without dimension-specific implementations.
struct K16FourRowShiftGroups {
    group_by_key: Vec<(u64, u8)>,
    group_columns: Vec<u8>,
    group_counts: Vec<u8>,
    group_shifts: Vec<[usize; 4]>,
    partial_columns: Vec<u8>,
    row_start: usize,
    num_groups: u8,
}

impl K16FourRowShiftGroups {
    fn new(num_columns: usize, row_start: usize) -> Option<Self> {
        if num_columns >= usize::from(u8::MAX) {
            return None;
        }
        let key_slots = (2 * num_columns).next_power_of_two();
        Some(Self {
            group_by_key: vec![(0, u8::MAX); key_slots],
            group_columns: vec![u8::MAX; num_columns * num_columns],
            group_counts: vec![0; num_columns],
            group_shifts: vec![[0; 4]; num_columns],
            partial_columns: Vec::with_capacity(num_columns),
            row_start,
            num_groups: 0,
        })
    }

    fn build(
        &mut self,
        selected_rows: &[u8],
        committed_zero_masks: &[u64],
        num_columns: usize,
    ) -> bool {
        self.group_by_key.fill((0, u8::MAX));
        self.partial_columns.clear();
        self.num_groups = 0;
        if selected_rows.len() != 4 * num_columns || committed_zero_masks.len() != 4 {
            return false;
        }

        for column in 0..num_columns {
            let mut key = 0u64;
            let mut shifts = [0usize; 4];
            let mut complete = true;
            for (row_offset, (row_indices, &committed_zero_mask)) in selected_rows
                .chunks_exact(num_columns)
                .zip(committed_zero_masks)
                .enumerate()
            {
                let hot = row_indices[column];
                if !row_is_committed(hot, committed_zero_mask, column) {
                    complete = false;
                    break;
                }
                key |= u64::from(hot) << (4 * row_offset);
                shifts[row_offset] = 16 * (self.row_start + row_offset) + usize::from(hot);
            }
            if !complete {
                self.partial_columns.push(column as u8);
                continue;
            }
            let slot_mask = self.group_by_key.len() - 1;
            let mut slot = key.wrapping_mul(0x9e37_79b9_7f4a_7c15) as usize & slot_mask;
            let group = loop {
                let (stored_key, stored_group) = self.group_by_key[slot];
                if stored_group != u8::MAX && stored_key == key {
                    break stored_group;
                }
                if stored_group == u8::MAX {
                    let group = self.num_groups;
                    self.num_groups += 1;
                    self.group_by_key[slot] = (key, group);
                    self.group_counts[usize::from(group)] = 0;
                    self.group_shifts[usize::from(group)] = shifts;
                    break group;
                }
                slot = (slot + 1) & slot_mask;
            };
            let group = usize::from(group);
            let count = usize::from(self.group_counts[group]);
            self.group_columns[group * num_columns + count] = column as u8;
            self.group_counts[group] += 1;
        }
        true
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the fused shift kernel keeps its source, destination, rank, and row views explicit"
    )]
    fn accumulate<const D: usize>(
        &self,
        src: &AkitaWideRing<D>,
        dst: &mut [AkitaWideRing<D>],
        a: usize,
        n_a: usize,
        selected_rows: &[u8],
        committed_zero_masks: &[u64],
        num_columns: usize,
    ) {
        for group in 0..self.num_groups {
            let group = usize::from(group);
            let count = usize::from(self.group_counts[group]);
            let columns = &self.group_columns[group * num_columns..group * num_columns + count];
            if self.group_counts[group] >= SHARED_SHIFT_MIN_COLUMNS {
                let mut shifted_sum = AkitaWideRing::zero();
                for &shift in &self.group_shifts[group] {
                    src.shift_accumulate_into(&mut shifted_sum, shift);
                }
                for &column in columns {
                    dst[usize::from(column) * n_a + a] += shifted_sum;
                }
            } else {
                for &column in columns {
                    let dst = &mut dst[usize::from(column) * n_a + a];
                    for &shift in &self.group_shifts[group] {
                        src.shift_accumulate_into(dst, shift);
                    }
                }
            }
        }
        for &column in &self.partial_columns {
            let column = usize::from(column);
            for (row_offset, (row_indices, &committed_zero_mask)) in selected_rows
                .chunks_exact(num_columns)
                .zip(committed_zero_masks)
                .enumerate()
            {
                let hot = row_indices[column];
                if row_is_committed(hot, committed_zero_mask, column) {
                    src.shift_accumulate_into(
                        &mut dst[column * n_a + a],
                        (self.row_start + row_offset) * 16 + usize::from(hot),
                    );
                }
            }
        }
    }
}

/// Row-major source for the semantic columns packed into `OneHotTrace`.
///
/// `fill_row` must overwrite all of `selected_rows`. Byte zero means no committed
/// entry unless [`TraceOneHotRows::committed_digit_zero_mask`] marks the column.
pub trait TraceOneHotRows: Send + Sync + 'static {
    fn num_rows(&self) -> usize;
    fn num_columns(&self) -> usize;
    fn fill_row(&self, row: usize, selected_rows: &mut [u8]);

    /// Bit `i` is set when column `i` commits row zero in this trace row.
    fn committed_digit_zero_mask(&self, _row: usize) -> u64 {
        0
    }

    /// Fills consecutive rows in row-major order, overwriting the entire buffer.
    fn fill_rows(&self, row_start: usize, selected_rows: &mut [u8]) {
        let num_columns = self.num_columns();
        debug_assert_eq!(selected_rows.len() % num_columns, 0);
        for (row_offset, row_indices) in selected_rows.chunks_exact_mut(num_columns).enumerate() {
            self.fill_row(row_start + row_offset, row_indices);
        }
    }

    /// Fills the masks for consecutive rows, overwriting the entire buffer.
    fn fill_committed_digit_zero_masks(&self, row_start: usize, masks: &mut [u64]) {
        for (row_offset, mask) in masks.iter_mut().enumerate() {
            *mask = self.committed_digit_zero_mask(row_start + row_offset);
        }
    }
}

/// Default value written by [`TraceOneHotRows::fill_row`] for an empty row.
#[must_use]
pub const fn no_selected_row() -> u8 {
    NO_SELECTED_ROW
}

/// One physical one-hot polynomial containing all trace-derived semantic
/// columns and zero padding up to a protocol-fixed selector capacity.
pub struct TracePackedOneHot {
    rows: Arc<RwLock<Option<Arc<dyn TraceOneHotRows>>>>,
    num_rows: usize,
    num_columns: usize,
    one_hot_k: usize,
    column_capacity: usize,
    num_vars: usize,
    construction_ring_elems: usize,
}

impl Clone for TracePackedOneHot {
    fn clone(&self) -> Self {
        let rows = self
            .rows
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        Self {
            rows: Arc::new(RwLock::new(rows)),
            num_rows: self.num_rows,
            num_columns: self.num_columns,
            one_hot_k: self.one_hot_k,
            column_capacity: self.column_capacity,
            num_vars: self.num_vars,
            construction_ring_elems: self.construction_ring_elems,
        }
    }
}

impl fmt::Debug for TracePackedOneHot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TracePackedOneHot")
            .field("one_hot_k", &self.one_hot_k)
            .field("num_columns", &self.num_columns)
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
        if !one_hot_k.is_power_of_two() || one_hot_k > 256 {
            return Err(AkitaError::InvalidInput(format!(
                "trace one-hot K={one_hot_k} must be a power of two fitting u8 row indices"
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
        if num_columns > u64::BITS as usize {
            return Err(AkitaError::InvalidInput(format!(
                "trace one-hot has {num_columns} semantic columns, above the 64-column mask limit"
            )));
        }
        if num_columns == 0 || num_columns > column_capacity {
            return Err(AkitaError::InvalidInput(format!(
                "trace one-hot has {num_columns} semantic columns for capacity {column_capacity}"
            )));
        }
        let num_rows = rows.num_rows();
        let total_field_elems = num_rows
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
            rows: Arc::new(RwLock::new(Some(rows))),
            num_rows,
            num_columns,
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
        let segment_field_elems = self.num_rows.checked_mul(self.one_hot_k).ok_or_else(|| {
            AkitaError::InvalidInput("trace one-hot segment ring count overflow".to_string())
        })?;
        if !segment_field_elems.is_multiple_of(D) {
            return Err(AkitaError::InvalidInput(format!(
                "trace one-hot semantic segment {segment_field_elems} is not ring-aligned at D={D}"
            )));
        }
        Ok(segment_field_elems / D)
    }

    fn lock_rows(
        &self,
    ) -> Result<RwLockReadGuard<'_, Option<Arc<dyn TraceOneHotRows>>>, AkitaError> {
        let rows = self
            .rows
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if rows.is_none() {
            return Err(AkitaError::InvalidInput(
                "trace one-hot opening storage was already released".to_string(),
            ));
        }
        Ok(rows)
    }
}

pub struct TracePackedOneHotView<'a, const D: usize> {
    source: &'a TracePackedOneHot,
    rows: RwLockReadGuard<'a, Option<Arc<dyn TraceOneHotRows>>>,
}

pub struct TracePackedOneHotBatchView<'a, const D: usize> {
    sources: &'a [&'a TracePackedOneHot],
    rows: RwLockReadGuard<'a, Option<Arc<dyn TraceOneHotRows>>>,
}

/// Borrowed root-source sum type used only by the heterogeneous
/// `[dense precommit, streamed trace final]` opening. Both variants borrow the
/// commit-time hint storage, so type erasure does not clone either source.
#[derive(Clone, Debug)]
pub(crate) enum GroupedRootSource {
    Dense(Arc<[DensePoly<AkitaField>]>),
    OneHot(Arc<[OneHotPoly<AkitaField, u8>]>),
    Trace(Arc<[TracePackedOneHot]>),
}

pub(crate) struct GroupedRootView<'view, const D: usize> {
    source: &'view GroupedRootSource,
}

pub(crate) struct GroupedRootBatchView<'view, const D: usize> {
    sources: &'view [&'view GroupedRootSource],
}

#[expect(
    clippy::panic,
    reason = "grouped root sources are constructed only after validating singleton hint storage"
)]
fn grouped_singleton<T>(values: &[T]) -> &T {
    let [value] = values else {
        panic!("grouped root source must retain exactly one polynomial")
    };
    value
}

impl RootPolyMeta<AkitaField> for GroupedRootSource {
    fn num_ring_elems(&self) -> usize {
        match self {
            Self::Dense(polys) => RootPolyMeta::num_ring_elems(grouped_singleton(polys)),
            Self::OneHot(polys) => RootPolyMeta::num_ring_elems(grouped_singleton(polys)),
            Self::Trace(polys) => RootPolyMeta::num_ring_elems(grouped_singleton(polys)),
        }
    }

    fn num_vars(&self) -> usize {
        match self {
            Self::Dense(polys) => RootPolyMeta::num_vars(grouped_singleton(polys)),
            Self::OneHot(polys) => RootPolyMeta::num_vars(grouped_singleton(polys)),
            Self::Trace(polys) => RootPolyMeta::num_vars(grouped_singleton(polys)),
        }
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        match self {
            Self::Dense(_) => None,
            Self::OneHot(polys) => RootPolyMeta::onehot_chunk_size(grouped_singleton(polys)),
            Self::Trace(polys) => RootPolyMeta::onehot_chunk_size(grouped_singleton(polys)),
        }
    }
}

impl<const D: usize> RootPolyShape<AkitaField, D> for GroupedRootSource {
    fn num_ring_elems(&self) -> usize {
        match self {
            Self::Dense(polys) => {
                RootPolyShape::<AkitaField, D>::num_ring_elems(grouped_singleton(polys))
            }
            Self::OneHot(polys) => {
                RootPolyShape::<AkitaField, D>::num_ring_elems(grouped_singleton(polys))
            }
            Self::Trace(polys) => {
                RootPolyShape::<AkitaField, D>::num_ring_elems(grouped_singleton(polys))
            }
        }
    }

    fn num_vars(&self) -> usize {
        match self {
            Self::Dense(polys) => {
                RootPolyShape::<AkitaField, D>::num_vars(grouped_singleton(polys))
            }
            Self::OneHot(polys) => {
                RootPolyShape::<AkitaField, D>::num_vars(grouped_singleton(polys))
            }
            Self::Trace(polys) => {
                RootPolyShape::<AkitaField, D>::num_vars(grouped_singleton(polys))
            }
        }
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        match self {
            Self::Dense(_) => None,
            Self::OneHot(polys) => {
                RootPolyShape::<AkitaField, D>::onehot_chunk_size(grouped_singleton(polys))
            }
            Self::Trace(polys) => {
                RootPolyShape::<AkitaField, D>::onehot_chunk_size(grouped_singleton(polys))
            }
        }
    }
}

impl<const D: usize> RootCommitSource<AkitaField, D> for GroupedRootSource {
    type CommitView<'view>
        = GroupedRootView<'view, D>
    where
        Self: 'view;

    fn commit_view(&self) -> Result<Self::CommitView<'_>, AkitaError> {
        Ok(GroupedRootView { source: self })
    }

    fn committed_centered_reach(
        &self,
        modulus: u128,
        centering_threshold: u128,
    ) -> Result<(u128, u128), AkitaError> {
        match self {
            Self::Dense(polys) => RootCommitSource::<AkitaField, D>::committed_centered_reach(
                grouped_singleton(polys),
                modulus,
                centering_threshold,
            ),
            Self::OneHot(polys) => RootCommitSource::<AkitaField, D>::committed_centered_reach(
                grouped_singleton(polys),
                modulus,
                centering_threshold,
            ),
            Self::Trace(polys) => RootCommitSource::<AkitaField, D>::committed_centered_reach(
                grouped_singleton(polys),
                modulus,
                centering_threshold,
            ),
        }
    }
}

impl<const D: usize> RootOpeningSource<AkitaField, D> for GroupedRootSource {
    type OpeningView<'view>
        = GroupedRootView<'view, D>
    where
        Self: 'view;
    type OpeningBatchView<'view>
        = GroupedRootBatchView<'view, D>
    where
        Self: 'view;

    fn opening_view(&self) -> Result<Self::OpeningView<'_>, AkitaError> {
        Ok(GroupedRootView { source: self })
    }

    fn opening_batch<'view>(
        polys: &'view [&'view Self],
    ) -> Result<Self::OpeningBatchView<'view>, AkitaError> {
        Ok(GroupedRootBatchView { sources: polys })
    }
}

impl<const D: usize> RootTensorSource<AkitaField, D> for GroupedRootSource {
    type TensorView<'view>
        = GroupedRootView<'view, D>
    where
        Self: 'view;
    type TensorBatchView<'view>
        = GroupedRootBatchView<'view, D>
    where
        Self: 'view;

    fn tensor_view(&self) -> Result<Self::TensorView<'_>, AkitaError> {
        Ok(GroupedRootView { source: self })
    }

    fn tensor_batch<'view>(
        polys: &'view [&'view Self],
    ) -> Result<Self::TensorBatchView<'view>, AkitaError> {
        Ok(GroupedRootBatchView { sources: polys })
    }
}

struct TracePackedOneHotKernelSource<'a> {
    source: &'a TracePackedOneHot,
    rows: &'a dyn TraceOneHotRows,
}

impl Deref for TracePackedOneHotKernelSource<'_> {
    type Target = TracePackedOneHot;

    fn deref(&self) -> &Self::Target {
        self.source
    }
}

impl<const D: usize> TracePackedOneHotView<'_, D> {
    fn kernel_source(&self) -> TracePackedOneHotKernelSource<'_> {
        let Some(rows) = self.rows.as_deref() else {
            unreachable!("trace one-hot view holds live row storage");
        };
        TracePackedOneHotKernelSource {
            source: self.source,
            rows,
        }
    }
}

impl<const D: usize> TracePackedOneHotBatchView<'_, D> {
    fn kernel_source(&self) -> TracePackedOneHotKernelSource<'_> {
        let Some(rows) = self.rows.as_deref() else {
            unreachable!("trace one-hot batch view holds live row storage");
        };
        TracePackedOneHotKernelSource {
            source: self.sources[0],
            rows,
        }
    }
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
        Ok(TracePackedOneHotView {
            source: self,
            rows: self.lock_rows()?,
        })
    }

    /// The packed trace stores hot positions, so every coefficient it commits is
    /// `0` or `1` and no scan is possible or needed.
    fn committed_centered_reach(
        &self,
        _modulus: u128,
        _centering_threshold: u128,
    ) -> Result<(u128, u128), AkitaError> {
        Ok((0, 1))
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
        Ok(TracePackedOneHotView {
            source: self,
            rows: self.lock_rows()?,
        })
    }

    fn opening_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::OpeningBatchView<'a>, AkitaError> {
        validate_singleton_batch(polys)?;
        validate_dimension::<D>(polys[0].one_hot_k)?;
        Ok(TracePackedOneHotBatchView {
            sources: polys,
            rows: polys[0].lock_rows()?,
        })
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
        Ok(TracePackedOneHotView {
            source: self,
            rows: self.lock_rows()?,
        })
    }

    fn tensor_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::TensorBatchView<'a>, AkitaError> {
        validate_singleton_batch(polys)?;
        validate_dimension::<D>(polys[0].one_hot_k)?;
        Ok(TracePackedOneHotBatchView {
            sources: polys,
            rows: polys[0].lock_rows()?,
        })
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
    source: &TracePackedOneHotKernelSource<'_>,
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
    let mut selected_rows = vec![NO_SELECTED_ROW; num_columns];
    if k >= D {
        let rings_per_row = k / D;
        let row_start = ring_start / rings_per_row;
        let row_end = ring_end.div_ceil(rings_per_row);
        let mut buckets = vec![Vec::new(); rings_per_row];
        for row in row_start..row_end.min(source.rows.num_rows()) {
            source.rows.fill_row(row, &mut selected_rows);
            let committed_zero_mask = source.rows.committed_digit_zero_mask(row);
            for bucket in &mut buckets {
                bucket.clear();
            }
            for (column, &hot) in selected_rows.iter().enumerate() {
                if !row_is_committed(hot, committed_zero_mask, column) {
                    continue;
                }
                let hot = usize::from(hot);
                if hot >= k {
                    return Err(AkitaError::InvalidInput(format!(
                        "trace one-hot row {hot} is outside K={k}"
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
                source.rows.fill_row(row, &mut selected_rows);
                let committed_zero_mask = source.rows.committed_digit_zero_mask(row);
                for (column, &hot) in selected_rows.iter().enumerate() {
                    if !row_is_committed(hot, committed_zero_mask, column) {
                        continue;
                    }
                    let hot = usize::from(hot);
                    if hot >= k {
                        return Err(AkitaError::InvalidInput(format!(
                            "trace one-hot row {hot} is outside K={k}"
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

/// Visits K<D ring elements as row indices for the D/K trace rows packed
/// into each ring. This avoids expanding the row buffer into contribution
/// tuples when a kernel can consume the indices directly.
fn visit_segment_ring_row_range<const D: usize>(
    source: &TracePackedOneHotKernelSource<'_>,
    ring_start: usize,
    ring_end: usize,
    mut visit: impl FnMut(usize, &[u8], &[u64]),
) -> Result<(), AkitaError> {
    validate_dimension::<D>(source.one_hot_k)?;
    if source.one_hot_k >= D {
        return Err(AkitaError::InvalidInput(format!(
            "trace one-hot row traversal requires K={} < D={D}",
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
    let row_index_count = num_columns.checked_mul(rows_per_ring).ok_or_else(|| {
        AkitaError::InvalidInput("trace one-hot row-index buffer size overflow".to_string())
    })?;
    let mut selected_rows = vec![NO_SELECTED_ROW; row_index_count];
    let mut committed_zero_masks = vec![0u64; rows_per_ring];
    for ring in ring_start..ring_end {
        let row_start = ring * rows_per_ring;
        let populated_rows = source
            .rows
            .num_rows()
            .saturating_sub(row_start)
            .min(rows_per_ring);
        let populated_indices = &mut selected_rows[..populated_rows * num_columns];
        let populated_masks = &mut committed_zero_masks[..populated_rows];
        source.rows.fill_rows(row_start, populated_indices);
        source
            .rows
            .fill_committed_digit_zero_masks(row_start, populated_masks);
        for &hot in populated_indices.iter() {
            if hot != NO_SELECTED_ROW && usize::from(hot) >= source.one_hot_k {
                return Err(AkitaError::InvalidInput(format!(
                    "trace one-hot row {hot} is outside K={}",
                    source.one_hot_k
                )));
            }
        }
        visit(ring, populated_indices, populated_masks);
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

fn flush_deferred_rank<const D: usize>(
    rank_deferred: &mut [DeferredFp128Ring<D>],
    reduced: &mut [CyclotomicRing<AkitaField, D>],
    n_a: usize,
    a: usize,
) {
    for (column, value) in rank_deferred.iter_mut().enumerate() {
        let index = column * n_a + a;
        reduced[index] += value.reduce_and_clear();
    }
}

#[inline(always)]
fn full_row_coefficients<const N: usize>(
    selected_rows: &[u8],
    committed_zero_masks: &[u64],
    num_columns: usize,
    column: usize,
    one_hot_k: usize,
) -> Option<[usize; N]> {
    if selected_rows.len() != N * num_columns || committed_zero_masks.len() != N {
        return None;
    }
    let coefficients = std::array::from_fn(|row| {
        row * one_hot_k + usize::from(selected_rows[row * num_columns + column])
    });
    (0..N)
        .all(|row| {
            row_is_committed(
                selected_rows[row * num_columns + column],
                committed_zero_masks[row],
                column,
            )
        })
        .then_some(coefficients)
}

#[inline(always)]
fn shift_accumulate_full_rows<const D: usize, const N: usize>(
    src: &AkitaWideRing<D>,
    dst: &mut AkitaWideRing<D>,
    selected_rows: &[u8],
    committed_zero_masks: &[u64],
    num_columns: usize,
    column: usize,
    one_hot_k: usize,
) -> bool {
    let Some(coefficients) = full_row_coefficients::<N>(
        selected_rows,
        committed_zero_masks,
        num_columns,
        column,
        one_hot_k,
    ) else {
        return false;
    };
    for coefficient in coefficients {
        src.shift_accumulate_into(dst, coefficient);
    }
    true
}

#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "the fixed-row fast path keeps its source, destination, row views, and geometry explicit"
)]
fn try_shift_accumulate_full_rows<const D: usize>(
    src: &AkitaWideRing<D>,
    dst: &mut AkitaWideRing<D>,
    selected_rows: &[u8],
    committed_zero_masks: &[u64],
    num_columns: usize,
    column: usize,
    one_hot_k: usize,
    rows_per_ring: usize,
) -> bool {
    match rows_per_ring {
        2 => shift_accumulate_full_rows::<D, 2>(
            src,
            dst,
            selected_rows,
            committed_zero_masks,
            num_columns,
            column,
            one_hot_k,
        ),
        4 => shift_accumulate_full_rows::<D, 4>(
            src,
            dst,
            selected_rows,
            committed_zero_masks,
            num_columns,
            column,
            one_hot_k,
        ),
        8 => shift_accumulate_full_rows::<D, 8>(
            src,
            dst,
            selected_rows,
            committed_zero_masks,
            num_columns,
            column,
            one_hot_k,
        ),
        16 => shift_accumulate_full_rows::<D, 16>(
            src,
            dst,
            selected_rows,
            committed_zero_masks,
            num_columns,
            column,
            one_hot_k,
        ),
        32 => shift_accumulate_full_rows::<D, 32>(
            src,
            dst,
            selected_rows,
            committed_zero_masks,
            num_columns,
            column,
            one_hot_k,
        ),
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

fn commit_packed<const D: usize>(
    backend: &CpuBackend,
    prepared: &<CpuBackend as ComputeBackendSetup<AkitaField>>::PreparedSetup,
    source: &TracePackedOneHotKernelSource<'_>,
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
                let rank_tiled_k256 = matches!(D, 64 | 128 | 256)
                    && source.one_hot_k == 256
                    && num_columns <= u32::BITS as usize;
                let mut wide = if rank_tiled_k256 {
                    Vec::new()
                } else {
                    vec![WideCyclotomicRing::zero(); num_columns * plan.n_a]
                };
                let mut budget = 0usize;
                if source.one_hot_k < D {
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

fn opening_fold_packed<const D: usize>(
    source: &TracePackedOneHotKernelSource<'_>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecomposeRotationMode {
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

fn decompose_fold_packed_with_mode<const D: usize>(
    source: &TracePackedOneHotKernelSource<'_>,
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

fn decompose_fold_packed<const D: usize>(
    source: &TracePackedOneHotKernelSource<'_>,
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
    fn commit_inner_group(
        &self,
        prepared: &Self::PreparedSetup,
        sources: Vec<TracePackedOneHotView<'_, D>>,
        plan: CommitInnerPlan,
    ) -> Result<Vec<CommitInnerWitness<AkitaField>>, AkitaError> {
        sources
            .into_par_iter()
            .map(|source| commit_packed::<D>(self, prepared, &source.kernel_source(), plan))
            .collect()
    }
}

impl<const D: usize> OpeningFoldKernel<TracePackedOneHotView<'_, D>, AkitaField, D> for CpuBackend {
    fn evaluate_and_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: TracePackedOneHotView<'_, D>,
        plan: OpeningFoldPlan<'_, AkitaField>,
    ) -> Result<OpeningFoldOutput<AkitaField, D>, AkitaError> {
        opening_fold_packed(&source.kernel_source(), plan)
    }

    fn decompose_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: TracePackedOneHotView<'_, D>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<akita_prover::DecomposeFoldWitness<AkitaField>, AkitaError> {
        decompose_fold_packed::<D>(
            &source.kernel_source(),
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
        let source = source.kernel_source();
        match plan {
            DecomposeFoldBatchPlan::Sparse {
                challenges,
                num_positions_per_block,
                num_digits,
                ..
            } => Ok(BatchDecomposeFoldOutcome::Fused(
                decompose_fold_packed::<D>(
                    &source,
                    challenges,
                    num_positions_per_block,
                    num_digits,
                )?,
            )),
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

impl<const D: usize> RootCommitKernel<GroupedRootView<'_, D>, AkitaField, D> for CpuBackend {
    fn commit_inner_group(
        &self,
        prepared: &Self::PreparedSetup,
        sources: Vec<GroupedRootView<'_, D>>,
        plan: CommitInnerPlan,
    ) -> Result<Vec<CommitInnerWitness<AkitaField>>, AkitaError> {
        let Some(first) = sources.first() else {
            return Err(AkitaError::InvalidInput(
                "grouped root commitment requires a nonempty group".to_string(),
            ));
        };
        match first.source {
            GroupedRootSource::Dense(_) => {
                let dense = sources
                    .into_iter()
                    .map(|source| match source.source {
                        GroupedRootSource::Dense(polys) => grouped_singleton(polys).commit_view(),
                        GroupedRootSource::OneHot(_) | GroupedRootSource::Trace(_) => {
                            Err(AkitaError::InvalidInput(
                                "grouped root commitment groups must be representation-homogeneous"
                                    .to_string(),
                            ))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                RootCommitKernel::<DenseView<'_, AkitaField, D>, AkitaField, D>::commit_inner_group(
                    self, prepared, dense, plan,
                )
            }
            GroupedRootSource::OneHot(_) => {
                let one_hot = sources
                    .into_iter()
                    .map(|source| match source.source {
                        GroupedRootSource::OneHot(polys) => grouped_singleton(polys).commit_view(),
                        GroupedRootSource::Dense(_) | GroupedRootSource::Trace(_) => {
                            Err(AkitaError::InvalidInput(
                                "grouped root commitment groups must be representation-homogeneous"
                                    .to_string(),
                            ))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                RootCommitKernel::<OneHotView<'_, AkitaField, D, u8>, AkitaField, D>::commit_inner_group(
                    self, prepared, one_hot, plan,
                )
            }
            GroupedRootSource::Trace(_) => {
                let trace = sources
                    .into_iter()
                    .map(|source| match source.source {
                        GroupedRootSource::Trace(polys) => grouped_singleton(polys).commit_view(),
                        GroupedRootSource::Dense(_) | GroupedRootSource::OneHot(_) => {
                            Err(AkitaError::InvalidInput(
                                "grouped root commitment groups must be representation-homogeneous"
                                    .to_string(),
                            ))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                RootCommitKernel::<TracePackedOneHotView<'_, D>, AkitaField, D>::commit_inner_group(
                    self, prepared, trace, plan,
                )
            }
        }
    }
}

impl<const D: usize> OpeningFoldKernel<GroupedRootView<'_, D>, AkitaField, D> for CpuBackend {
    fn evaluate_and_fold(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: GroupedRootView<'_, D>,
        plan: OpeningFoldPlan<'_, AkitaField>,
    ) -> Result<OpeningFoldOutput<AkitaField, D>, AkitaError> {
        match source.source {
            GroupedRootSource::Dense(polys) => {
                OpeningFoldKernel::<DenseView<'_, AkitaField, D>, AkitaField, D>::evaluate_and_fold(
                    self,
                    prepared,
                    grouped_singleton(polys).opening_view()?,
                    plan,
                )
            }
            GroupedRootSource::OneHot(polys) => OpeningFoldKernel::<
                OneHotView<'_, AkitaField, D, u8>,
                AkitaField,
                D,
            >::evaluate_and_fold(
                self,
                prepared,
                grouped_singleton(polys).opening_view()?,
                plan,
            ),
            GroupedRootSource::Trace(polys) => {
                OpeningFoldKernel::<TracePackedOneHotView<'_, D>, AkitaField, D>::evaluate_and_fold(
                    self,
                    prepared,
                    grouped_singleton(polys).opening_view()?,
                    plan,
                )
            }
        }
    }

    fn decompose_fold(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: GroupedRootView<'_, D>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<akita_prover::DecomposeFoldWitness<AkitaField>, AkitaError> {
        match source.source {
            GroupedRootSource::Dense(polys) => {
                OpeningFoldKernel::<DenseView<'_, AkitaField, D>, AkitaField, D>::decompose_fold(
                    self,
                    prepared,
                    grouped_singleton(polys).opening_view()?,
                    plan,
                )
            }
            GroupedRootSource::OneHot(polys) => OpeningFoldKernel::<
                OneHotView<'_, AkitaField, D, u8>,
                AkitaField,
                D,
            >::decompose_fold(
                self,
                prepared,
                grouped_singleton(polys).opening_view()?,
                plan,
            ),
            GroupedRootSource::Trace(polys) => {
                OpeningFoldKernel::<TracePackedOneHotView<'_, D>, AkitaField, D>::decompose_fold(
                    self,
                    prepared,
                    grouped_singleton(polys).opening_view()?,
                    plan,
                )
            }
        }
    }
}

impl<const D: usize> OpeningBatchKernel<GroupedRootBatchView<'_, D>, AkitaField, D> for CpuBackend {
    fn decompose_fold_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: GroupedRootBatchView<'_, D>,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<BatchDecomposeFoldOutcome<AkitaField, D>, AkitaError> {
        let Some(first) = source.sources.first() else {
            return Ok(BatchDecomposeFoldOutcome::FallbackPerPoly);
        };
        match first {
            GroupedRootSource::Dense(_) => {
                let dense = source
                    .sources
                    .iter()
                    .map(|source| match source {
                        GroupedRootSource::Dense(polys) => Ok(grouped_singleton(polys)),
                        GroupedRootSource::OneHot(_) | GroupedRootSource::Trace(_) => {
                            Err(AkitaError::InvalidInput(
                                "grouped root opening groups must be representation-homogeneous"
                                    .to_string(),
                            ))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let view =
                    <DensePoly<AkitaField> as RootOpeningSource<AkitaField, D>>::opening_batch(
                        &dense,
                    )?;
                OpeningBatchKernel::<DenseBatchView<'_, AkitaField, D>, AkitaField, D>::decompose_fold_batch(
                    self, prepared, view, plan,
                )
            }
            GroupedRootSource::OneHot(_) => {
                let one_hot = source
                    .sources
                    .iter()
                    .map(|source| match source {
                        GroupedRootSource::OneHot(polys) => Ok(grouped_singleton(polys)),
                        GroupedRootSource::Dense(_) | GroupedRootSource::Trace(_) => {
                            Err(AkitaError::InvalidInput(
                                "grouped root opening groups must be representation-homogeneous"
                                    .to_string(),
                            ))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let view = <OneHotPoly<AkitaField, u8> as RootOpeningSource<
                    AkitaField,
                    D,
                >>::opening_batch(&one_hot)?;
                OpeningBatchKernel::<
                    OneHotBatchView<'_, AkitaField, D, u8>,
                    AkitaField,
                    D,
                >::decompose_fold_batch(self, prepared, view, plan)
            }
            GroupedRootSource::Trace(_) => {
                let trace = source
                    .sources
                    .iter()
                    .map(|source| match source {
                        GroupedRootSource::Trace(polys) => Ok(grouped_singleton(polys)),
                        GroupedRootSource::Dense(_) | GroupedRootSource::OneHot(_) => {
                            Err(AkitaError::InvalidInput(
                                "grouped root opening groups must be representation-homogeneous"
                                    .to_string(),
                            ))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let view =
                    <TracePackedOneHot as RootOpeningSource<AkitaField, D>>::opening_batch(&trace)?;
                OpeningBatchKernel::<TracePackedOneHotBatchView<'_, D>, AkitaField, D>::decompose_fold_batch(
                    self, prepared, view, plan,
                )
            }
        }
    }
}

impl<E, const D: usize> TensorProjectionKernel<GroupedRootView<'_, D>, AkitaField, E, D>
    for CpuBackend
where
    E: ExtField<AkitaField>,
{
    fn column_partials(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: GroupedRootView<'_, D>,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<AkitaField>,
    {
        match source.source {
            GroupedRootSource::Dense(polys) => TensorProjectionKernel::<
                DenseView<'_, AkitaField, D>,
                AkitaField,
                E,
                D,
            >::column_partials(
                self,
                prepared,
                grouped_singleton(polys).tensor_view()?,
                logical_point,
            ),
            GroupedRootSource::OneHot(polys) => TensorProjectionKernel::<
                OneHotView<'_, AkitaField, D, u8>,
                AkitaField,
                E,
                D,
            >::column_partials(
                self,
                prepared,
                grouped_singleton(polys).tensor_view()?,
                logical_point,
            ),
            GroupedRootSource::Trace(polys) => TensorProjectionKernel::<
                TracePackedOneHotView<'_, D>,
                AkitaField,
                E,
                D,
            >::column_partials(
                self,
                prepared,
                grouped_singleton(polys).tensor_view()?,
                logical_point,
            ),
        }
    }

    fn packed_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: GroupedRootView<'_, D>,
    ) -> Result<TensorPackedWitness<E>, AkitaError> {
        match source.source {
            GroupedRootSource::Dense(polys) => TensorProjectionKernel::<
                DenseView<'_, AkitaField, D>,
                AkitaField,
                E,
                D,
            >::packed_witness(
                self,
                prepared,
                grouped_singleton(polys).tensor_view()?,
            ),
            GroupedRootSource::OneHot(polys) => TensorProjectionKernel::<
                OneHotView<'_, AkitaField, D, u8>,
                AkitaField,
                E,
                D,
            >::packed_witness(
                self,
                prepared,
                grouped_singleton(polys).tensor_view()?,
            ),
            GroupedRootSource::Trace(polys) => TensorProjectionKernel::<
                TracePackedOneHotView<'_, D>,
                AkitaField,
                E,
                D,
            >::packed_witness(
                self,
                prepared,
                grouped_singleton(polys).tensor_view()?,
            ),
        }
    }

    fn root_projection(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: GroupedRootView<'_, D>,
    ) -> Result<RootTensorProjectionPoly<AkitaField>, AkitaError>
    where
        AkitaField: FromPrimitiveInt,
        E: FpExtEncoding<AkitaField>,
    {
        match source.source {
            GroupedRootSource::Dense(polys) => TensorProjectionKernel::<
                DenseView<'_, AkitaField, D>,
                AkitaField,
                E,
                D,
            >::root_projection(
                self,
                prepared,
                grouped_singleton(polys).tensor_view()?,
            ),
            GroupedRootSource::OneHot(polys) => TensorProjectionKernel::<
                OneHotView<'_, AkitaField, D, u8>,
                AkitaField,
                E,
                D,
            >::root_projection(
                self,
                prepared,
                grouped_singleton(polys).tensor_view()?,
            ),
            GroupedRootSource::Trace(polys) => TensorProjectionKernel::<
                TracePackedOneHotView<'_, D>,
                AkitaField,
                E,
                D,
            >::root_projection(
                self,
                prepared,
                grouped_singleton(polys).tensor_view()?,
            ),
        }
    }
}

impl<E, const D: usize> TensorProjectionBatchKernel<GroupedRootBatchView<'_, D>, AkitaField, E, D>
    for CpuBackend
where
    E: ExtField<AkitaField>,
{
    fn column_partials_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: GroupedRootBatchView<'_, D>,
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<AkitaField>,
    {
        let Some(first) = source.sources.first() else {
            return Ok(Vec::new());
        };
        match first {
            GroupedRootSource::Dense(_) => {
                let dense = source
                    .sources
                    .iter()
                    .map(|source| match source {
                        GroupedRootSource::Dense(polys) => Ok(grouped_singleton(polys)),
                        GroupedRootSource::OneHot(_) | GroupedRootSource::Trace(_) => {
                            Err(AkitaError::InvalidInput(
                                "grouped root tensor groups must be representation-homogeneous"
                                    .to_string(),
                            ))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let view =
                    <DensePoly<AkitaField> as RootTensorSource<AkitaField, D>>::tensor_batch(
                        &dense,
                    )?;
                TensorProjectionBatchKernel::<
                    DenseBatchView<'_, AkitaField, D>,
                    AkitaField,
                    E,
                    D,
                >::column_partials_batch(self, prepared, view, logical_point)
            }
            GroupedRootSource::OneHot(_) => {
                let one_hot = source
                    .sources
                    .iter()
                    .map(|source| match source {
                        GroupedRootSource::OneHot(polys) => Ok(grouped_singleton(polys)),
                        GroupedRootSource::Dense(_) | GroupedRootSource::Trace(_) => {
                            Err(AkitaError::InvalidInput(
                                "grouped root tensor groups must be representation-homogeneous"
                                    .to_string(),
                            ))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let view =
                    <OneHotPoly<AkitaField, u8> as RootTensorSource<AkitaField, D>>::tensor_batch(
                        &one_hot,
                    )?;
                TensorProjectionBatchKernel::<
                    OneHotBatchView<'_, AkitaField, D, u8>,
                    AkitaField,
                    E,
                    D,
                >::column_partials_batch(self, prepared, view, logical_point)
            }
            GroupedRootSource::Trace(_) => {
                let trace = source
                    .sources
                    .iter()
                    .map(|source| match source {
                        GroupedRootSource::Trace(polys) => Ok(grouped_singleton(polys)),
                        GroupedRootSource::Dense(_) | GroupedRootSource::OneHot(_) => {
                            Err(AkitaError::InvalidInput(
                                "grouped root tensor groups must be representation-homogeneous"
                                    .to_string(),
                            ))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let view =
                    <TracePackedOneHot as RootTensorSource<AkitaField, D>>::tensor_batch(&trace)?;
                TensorProjectionBatchKernel::<
                    TracePackedOneHotBatchView<'_, D>,
                    AkitaField,
                    E,
                    D,
                >::column_partials_batch(self, prepared, view, logical_point)
            }
        }
    }

    fn sparse_linear_combination(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: GroupedRootBatchView<'_, D>,
        coeffs: &[E],
    ) -> Result<
        Option<
            akita_prover::protocol::extension_opening_reduction::SparseExtensionOpeningWitness<E>,
        >,
        AkitaError,
    > {
        let Some(first) = source.sources.first() else {
            return Ok(None);
        };
        match first {
            GroupedRootSource::Dense(_) => {
                let dense = source
                    .sources
                    .iter()
                    .map(|source| match source {
                        GroupedRootSource::Dense(polys) => Ok(grouped_singleton(polys)),
                        GroupedRootSource::OneHot(_) | GroupedRootSource::Trace(_) => {
                            Err(AkitaError::InvalidInput(
                                "grouped root tensor groups must be representation-homogeneous"
                                    .to_string(),
                            ))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let view =
                    <DensePoly<AkitaField> as RootTensorSource<AkitaField, D>>::tensor_batch(
                        &dense,
                    )?;
                TensorProjectionBatchKernel::<
                    DenseBatchView<'_, AkitaField, D>,
                    AkitaField,
                    E,
                    D,
                >::sparse_linear_combination(self, prepared, view, coeffs)
            }
            GroupedRootSource::OneHot(_) => {
                let one_hot = source
                    .sources
                    .iter()
                    .map(|source| match source {
                        GroupedRootSource::OneHot(polys) => Ok(grouped_singleton(polys)),
                        GroupedRootSource::Dense(_) | GroupedRootSource::Trace(_) => {
                            Err(AkitaError::InvalidInput(
                                "grouped root tensor groups must be representation-homogeneous"
                                    .to_string(),
                            ))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let view =
                    <OneHotPoly<AkitaField, u8> as RootTensorSource<AkitaField, D>>::tensor_batch(
                        &one_hot,
                    )?;
                TensorProjectionBatchKernel::<
                    OneHotBatchView<'_, AkitaField, D, u8>,
                    AkitaField,
                    E,
                    D,
                >::sparse_linear_combination(self, prepared, view, coeffs)
            }
            GroupedRootSource::Trace(_) => {
                let trace = source
                    .sources
                    .iter()
                    .map(|source| match source {
                        GroupedRootSource::Trace(polys) => Ok(grouped_singleton(polys)),
                        GroupedRootSource::Dense(_) | GroupedRootSource::OneHot(_) => {
                            Err(AkitaError::InvalidInput(
                                "grouped root tensor groups must be representation-homogeneous"
                                    .to_string(),
                            ))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let view =
                    <TracePackedOneHot as RootTensorSource<AkitaField, D>>::tensor_batch(&trace)?;
                TensorProjectionBatchKernel::<
                    TracePackedOneHotBatchView<'_, D>,
                    AkitaField,
                    E,
                    D,
                >::sparse_linear_combination(self, prepared, view, coeffs)
            }
        }
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
        committed_zero_column: Option<usize>,
    }

    impl TestRows {
        fn selected_row(&self, row: usize, column: usize) -> u8 {
            ((row * (2 * column + 1) + column) % self.k) as u8
        }
    }

    impl TraceOneHotRows for TestRows {
        fn num_rows(&self) -> usize {
            self.rows
        }

        fn num_columns(&self) -> usize {
            self.columns
        }

        fn fill_row(&self, row: usize, selected_rows: &mut [u8]) {
            for (column, selected) in selected_rows.iter_mut().enumerate() {
                *selected = self.selected_row(row, column);
            }
        }

        fn committed_digit_zero_mask(&self, row: usize) -> u64 {
            self.committed_zero_column
                .filter(|&column| self.selected_row(row, column) == 0)
                .map_or(0, |column| 1u64 << column)
        }
    }

    fn assert_ring_mapping<const D: usize>(
        k: usize,
        rows: usize,
        committed_zero_column: Option<usize>,
    ) {
        let source = TracePackedOneHot::new(
            k,
            64,
            8,
            Arc::new(TestRows {
                rows,
                columns: 3,
                k,
                committed_zero_column,
            }),
        )
        .unwrap();
        let segment_rings = source.segment_ring_elems::<D>().unwrap();
        let mut actual = Vec::new();
        let view =
            <TracePackedOneHot as RootOpeningSource<AkitaField, D>>::opening_view(&source).unwrap();
        visit_segment_ring_range::<D>(
            &view.kernel_source(),
            0,
            segment_rings,
            |ring, contributions| {
                actual.extend(
                    contributions
                        .iter()
                        .map(|&(column, coefficient)| (column, ring * D + coefficient)),
                );
            },
        )
        .unwrap();
        let expected = (0..rows)
            .flat_map(|row| {
                (0..3).filter_map(move |column| {
                    let selected_row = (row * (2 * column + 1) + column) % k;
                    (selected_row != 0 || committed_zero_column == Some(column))
                        .then_some((column, row * k + selected_row))
                })
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
            assert_ring_mapping::<64>(16, rows, None);
            assert_ring_mapping::<128>(16, rows, None);
            assert_ring_mapping::<256>(16, rows, None);
            assert_ring_mapping::<512>(16, rows, None);
            assert_ring_mapping::<64>(256, rows, None);
            assert_ring_mapping::<128>(256, rows, None);
            assert_ring_mapping::<256>(256, rows, None);
            assert_ring_mapping::<512>(256, rows, None);
        }
    }

    #[test]
    fn committed_digit_zero_mapping_is_dimension_generic() {
        assert_ring_mapping::<64>(16, 32, Some(1));
        assert_ring_mapping::<64>(256, 32, Some(1));
    }

    fn assert_k16_shift_groups<const D: usize>() {
        const COLUMNS: usize = 5;
        let rows_per_ring = D / 16;
        let mut selected_rows = vec![NO_SELECTED_ROW; rows_per_ring * COLUMNS];
        let committed_zero_masks = vec![0u64; rows_per_ring];
        for row in 0..rows_per_ring {
            let shared_hot = ((row + 1) % 15 + 1) as u8;
            selected_rows[row * COLUMNS] = shared_hot;
            selected_rows[row * COLUMNS + 1] = shared_hot;
            selected_rows[row * COLUMNS + 2] = shared_hot;
            selected_rows[row * COLUMNS + 3] = ((2 * row + 3) % 15 + 1) as u8;
            selected_rows[row * COLUMNS + 4] = if row == 1 {
                NO_SELECTED_ROW
            } else {
                ((3 * row + 5) % 15 + 1) as u8
            };
        }

        let source: CyclotomicRing<AkitaField, D> =
            CyclotomicRing::from_coefficients(std::array::from_fn(|index| {
                AkitaField::from_u64((index + 1) as u64)
            }));
        let source: AkitaWideRing<D> = AkitaWideRing::from_ring(&source);
        let mut actual = vec![AkitaWideRing::zero(); COLUMNS];
        for (chunk, chunk_rows) in selected_rows.chunks_exact(4 * COLUMNS).enumerate() {
            let masks = &committed_zero_masks[4 * chunk..4 * chunk + 4];
            let mut groups = K16FourRowShiftGroups::new(COLUMNS, 4 * chunk).unwrap();
            assert!(groups.build(chunk_rows, masks, COLUMNS));
            groups.accumulate(&source, &mut actual, 0, 1, chunk_rows, masks, COLUMNS);
        }

        let mut expected = vec![AkitaWideRing::zero(); COLUMNS];
        for (row, row_indices) in selected_rows.chunks_exact(COLUMNS).enumerate() {
            for (column, &hot) in row_indices.iter().enumerate() {
                if hot != NO_SELECTED_ROW {
                    source
                        .shift_accumulate_into(&mut expected[column], 16 * row + usize::from(hot));
                }
            }
        }
        let actual = actual
            .into_iter()
            .map(|value| value.reduce::<AkitaField>())
            .collect::<Vec<_>>();
        let expected = expected
            .into_iter()
            .map(|value| value.reduce::<AkitaField>())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn k16_shared_shift_groups_cover_adaptive_dimensions() {
        assert_k16_shift_groups::<64>();
        assert_k16_shift_groups::<128>();
        assert_k16_shift_groups::<256>();
    }

    #[test]
    fn constructor_enforces_selector_capacity() {
        let rows = Arc::new(TestRows {
            rows: 32,
            columns: 9,
            k: 16,
            committed_zero_column: None,
        });
        assert!(TracePackedOneHot::new(16, 64, 8, rows).is_err());
    }

    fn assert_deferred_fp128_shift_accumulator<const D: usize>() {
        let source: CyclotomicRing<AkitaField, D> =
            CyclotomicRing::from_coefficients(std::array::from_fn(|_| -AkitaField::one()));
        let mut expected: CyclotomicRing<AkitaField, D> = CyclotomicRing::zero();
        let mut deferred: DeferredFp128Ring<D> = DeferredFp128Ring::zero();

        for _ in 0..K256_ROW_BATCH {
            source.shift_accumulate_into(&mut expected, D / 2);
            deferred.shift_accumulate(&source, D / 2);
        }

        assert!(deferred
            .wraps
            .iter()
            .all(|wraps| usize::from(wraps.unsigned_abs()) <= K256_ROW_BATCH));
        assert_eq!(deferred.reduce_and_clear(), expected);
        assert!(deferred.lo.iter().all(|&limb| limb == 0));
        assert!(deferred.hi.iter().all(|&limb| limb == 0));
        assert!(deferred.wraps.iter().all(|&wraps| wraps == 0));

        let mut expected_after_reuse = CyclotomicRing::zero();
        source.shift_accumulate_into(&mut expected_after_reuse, D - 1);
        deferred.shift_accumulate(&source, D - 1);
        assert_eq!(deferred.reduce_and_clear(), expected_after_reuse);
        assert_eq!(std::mem::size_of::<DeferredFp128Ring<D>>(), 18 * D);
    }

    #[test]
    fn deferred_fp128_shift_accumulator_matches_canonical_at_batch_bound() {
        assert_deferred_fp128_shift_accumulator::<64>();
        assert_deferred_fp128_shift_accumulator::<128>();
        assert_deferred_fp128_shift_accumulator::<256>();
    }

    fn assert_opening_kernels_match_materialized<const D: usize>(
        k: usize,
        rows: usize,
        num_positions: usize,
        committed_zero_column: Option<usize>,
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
                committed_zero_column,
            }),
        )
        .unwrap();
        let packed_indices = (0..CAPACITY)
            .flat_map(|column| {
                (0..rows).map(move |row| {
                    let selected_row = ((row * (2 * column + 1) + column) % k) as u8;
                    (column < COLUMNS
                        && (selected_row != 0 || committed_zero_column == Some(column)))
                    .then_some(selected_row)
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
        let backend = CpuBackend::DEFAULT;
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
                positions: vec![0, (block % (D - 1) + 1) as u32].into(),
                coeffs: vec![1, -1].into(),
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
        let view =
            <TracePackedOneHot as RootOpeningSource<AkitaField, D>>::opening_view(&source).unwrap();
        let source = view.kernel_source();
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
        let compact = decompose_fold_packed_with_mode::<D>(
            &source,
            &challenges,
            num_positions,
            2,
            DecomposeRotationMode::Compact,
        )
        .unwrap();
        assert_eq!(dense, materialized);
        assert_eq!(sparse, materialized);
        assert_eq!(compact, materialized);
    }

    #[test]
    fn d64_k256_2e28_dense_rotations_fit_the_table_budget() {
        let table_bytes = 1024 * 29 * 64 * std::mem::size_of::<[i16; 64]>();
        assert_eq!(table_bytes, 243_269_632);
        assert!(table_bytes <= ROTATED_CHALLENGE_TABLE_BUDGET);
    }

    #[test]
    fn d128_auto_uses_compact_rotations() {
        let challenges = [SparseChallenge {
            positions: vec![0, 127].into(),
            coeffs: vec![1, -1].into(),
        }];
        let rotations =
            prepare_rotations::<128>(&challenges, None, 1, DecomposeRotationMode::Auto).unwrap();
        assert!(matches!(rotations, PreparedRotations::Compact(_)));
    }

    #[test]
    fn blockwise_opening_kernels_match_materialized_onehot() {
        assert_opening_kernels_match_materialized::<64>(256, 32, 16, None);
        assert_opening_kernels_match_materialized::<128>(256, 32, 16, None);
        assert_opening_kernels_match_materialized::<256>(256, 32, 16, None);
        assert_opening_kernels_match_materialized::<512>(256, 32, 8, None);
        assert_opening_kernels_match_materialized::<64>(16, 32, 4, None);
        assert_opening_kernels_match_materialized::<128>(16, 32, 2, None);
        assert_opening_kernels_match_materialized::<256>(16, 32, 2, None);
        assert_opening_kernels_match_materialized::<512>(16, 32, 1, None);
        assert_opening_kernels_match_materialized::<64>(16, 32, 16, None);
        assert_opening_kernels_match_materialized::<128>(16, 32, 8, None);
        assert_opening_kernels_match_materialized::<256>(16, 32, 4, None);
        assert_opening_kernels_match_materialized::<512>(16, 32, 2, None);
        assert_opening_kernels_match_materialized::<64>(256, 32, 16, Some(1));
        assert_opening_kernels_match_materialized::<64>(16, 32, 4, Some(1));
    }
}
