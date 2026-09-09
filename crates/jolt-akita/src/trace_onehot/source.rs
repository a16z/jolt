use std::{
    fmt::{Debug, Formatter, Result as FmtResult},
    sync::Arc,
};

use akita_error::AkitaError;
use akita_prover::{RootCommitSource, RootOpeningSource, RootPolyMeta, RootPolyShape};

use super::NO_SELECTED_ROW;
use crate::AkitaField;

/// Borrowed row-major selectors plus optional precomputed metrics for a
/// resident trace source; the Metal commit and opening paths consume them.
#[derive(Clone, Copy, Debug)]
pub struct TracePackedSelectors<'a> {
    row_major: &'a [u8],
    active_zero_rows: &'a [u64],
    zero_column_mask: u64,
    hot_entries: Option<usize>,
    zero_suffix_start: Option<usize>,
}

impl<'a> TracePackedSelectors<'a> {
    #[must_use]
    pub fn new(row_major: &'a [u8], active_zero_rows: &'a [u64], zero_column_mask: u64) -> Self {
        Self {
            row_major,
            active_zero_rows,
            zero_column_mask,
            hot_entries: None,
            zero_suffix_start: None,
        }
    }

    #[must_use]
    pub fn new_with_hot_entries(
        row_major: &'a [u8],
        active_zero_rows: &'a [u64],
        zero_column_mask: u64,
        hot_entries: usize,
    ) -> Self {
        Self {
            row_major,
            active_zero_rows,
            zero_column_mask,
            hot_entries: Some(hot_entries),
            zero_suffix_start: None,
        }
    }

    #[must_use]
    pub fn new_with_precomputed_metrics(
        row_major: &'a [u8],
        active_zero_rows: &'a [u64],
        zero_column_mask: u64,
        hot_entries: usize,
        zero_suffix_start: usize,
    ) -> Self {
        Self {
            row_major,
            active_zero_rows,
            zero_column_mask,
            hot_entries: Some(hot_entries),
            zero_suffix_start: Some(zero_suffix_start),
        }
    }

    #[must_use]
    pub fn row_major(self) -> &'a [u8] {
        self.row_major
    }

    #[must_use]
    pub fn active_zero_rows(self) -> &'a [u64] {
        self.active_zero_rows
    }

    #[must_use]
    pub fn zero_column_mask(self) -> u64 {
        self.zero_column_mask
    }

    #[must_use]
    pub fn hot_entries(self) -> Option<usize> {
        self.hot_entries
    }

    #[must_use]
    pub fn zero_suffix_start(self) -> Option<usize> {
        self.zero_suffix_start
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

    /// Borrows a resident row-major representation when the source has one.
    fn packed_selectors(&self) -> Option<TracePackedSelectors<'_>> {
        None
    }

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
#[derive(Clone)]
pub struct TracePackedOneHot {
    pub(super) rows: Arc<dyn TraceOneHotRows>,
    pub(super) num_rows: usize,
    pub(super) num_columns: usize,
    pub(super) one_hot_k: usize,
    pub(super) column_capacity: usize,
    pub(super) num_vars: usize,
}

impl Debug for TracePackedOneHot {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
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
            rows,
            num_rows,
            num_columns,
            one_hot_k,
            column_capacity,
            num_vars: total_field_elems.trailing_zeros() as usize,
        })
    }

    pub(super) fn total_field_elems(&self) -> usize {
        1usize << self.num_vars
    }

    pub(super) fn segment_ring_elems<const D: usize>(&self) -> Result<usize, AkitaError> {
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
}

pub struct TracePackedOneHotView<'a, const D: usize> {
    pub(super) source: &'a TracePackedOneHot,
}

pub struct TracePackedOneHotBatchView<'a, const D: usize> {
    pub(super) sources: &'a [&'a TracePackedOneHot],
}

impl<const D: usize> TracePackedOneHotView<'_, D> {
    pub(super) fn source(&self) -> &TracePackedOneHot {
        self.source
    }
}

impl<const D: usize> TracePackedOneHotBatchView<'_, D> {
    pub(super) fn source(&self) -> &TracePackedOneHot {
        self.sources[0]
    }
}

impl RootPolyMeta<AkitaField> for TracePackedOneHot {
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
        Ok(TracePackedOneHotView { source: self })
    }

    fn opening_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::OpeningBatchView<'a>, AkitaError> {
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

pub(super) fn validate_dimension<const D: usize>(one_hot_k: usize) -> Result<(), AkitaError> {
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
