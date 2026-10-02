use std::{
    borrow::Cow,
    fmt::{Debug, Formatter, Result as FmtResult},
    sync::Arc,
};

use akita_error::AkitaError;
use akita_pcs::custom_source::{
    AvailablePolynomialTypes, BackendKindId, CommitInnerPlan, CommitSourceClass,
    CommitSourceDescriptor, CommitmentSource, ExternalInnerCommitmentCapability,
    PolynomialRepresentation, PolynomialTypeSelection, PreparedExternalInnerCommitment,
    RootOpeningSource, RootPolyMeta, RootPolyShape, SourceCoefficients,
};

use super::kernels::{trace_commitment_capability, TraceOneHotColumnCommitOperation};
use super::NO_SELECTED_ROW;
use crate::AkitaField;

/// Row-major source for the native columns in `OneHotTrace`.
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

/// A native column view retaining the shared trace owner.
#[derive(Clone)]
pub struct TraceOneHotColumn {
    pub(super) rows: Arc<dyn TraceOneHotRows>,
    pub(super) num_rows: usize,
    pub(super) num_columns: usize,
    pub(super) one_hot_k: usize,
    pub(super) column_index: usize,
    pub(super) num_vars: usize,
}

impl Debug for TraceOneHotColumn {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("TraceOneHotColumn")
            .field("one_hot_k", &self.one_hot_k)
            .field("num_columns", &self.num_columns)
            .field("column_index", &self.column_index)
            .field("num_vars", &self.num_vars)
            .finish_non_exhaustive()
    }
}

impl TraceOneHotColumn {
    /// Constructs the ordered native column views.
    ///
    /// `construction_ring_d` is metadata matching the configured Akita
    /// commitment dimension. Kernel views remain const-generic over `D`.
    pub fn new(
        one_hot_k: usize,
        construction_ring_d: usize,
        rows: Arc<dyn TraceOneHotRows>,
    ) -> Result<Vec<Self>, AkitaError> {
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
        let num_columns = rows.num_columns();
        if num_columns > u64::BITS as usize {
            return Err(AkitaError::InvalidInput(format!(
                "trace one-hot has {num_columns} semantic columns, above the 64-column mask limit"
            )));
        }
        if num_columns == 0 {
            return Err(AkitaError::InvalidInput(
                "trace one-hot has no columns".into(),
            ));
        }
        let num_rows = rows.num_rows();
        let total_field_elems = num_rows.checked_mul(one_hot_k).ok_or_else(|| {
            AkitaError::InvalidInput("trace one-hot column domain overflow".to_string())
        })?;
        if !total_field_elems.is_power_of_two()
            || !total_field_elems.is_multiple_of(construction_ring_d)
        {
            return Err(AkitaError::InvalidInput(format!(
                "trace one-hot column domain {total_field_elems} must be a power of two divisible by construction D={construction_ring_d}"
            )));
        }
        Ok((0..num_columns)
            .map(|column_index| Self {
                rows: Arc::clone(&rows),
                num_rows,
                num_columns,
                one_hot_k,
                column_index,
                num_vars: total_field_elems.trailing_zeros() as usize,
            })
            .collect())
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

pub struct TraceOneHotColumnView<'a, const D: usize> {
    pub(super) source: &'a TraceOneHotColumn,
}

pub struct TraceOneHotColumnBatchView<'a, const D: usize> {
    pub(super) sources: &'a [&'a TraceOneHotColumn],
}

impl<const D: usize> TraceOneHotColumnView<'_, D> {
    pub(super) fn source(&self) -> &TraceOneHotColumn {
        self.source
    }
}

impl<const D: usize> TraceOneHotColumnBatchView<'_, D> {
    pub(super) fn source(&self) -> &TraceOneHotColumn {
        self.sources[0]
    }
}

impl RootPolyMeta<AkitaField> for TraceOneHotColumn {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        Some(self.one_hot_k)
    }
}

impl<const D: usize> RootPolyShape<AkitaField, D> for TraceOneHotColumn {
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

/// The trace streams hot positions from its rows. The canonical
/// coefficient table feeds only tensor-style extension openings, which
/// Jolt's base-field configs (`ExtField = Field`) never schedule.
impl SourceCoefficients<AkitaField> for TraceOneHotColumn {
    fn source_coefficients(&self) -> Result<Cow<'_, [AkitaField]>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "trace one-hot sources stream their coefficients and expose no canonical table".into(),
        ))
    }
}

impl CommitmentSource<AkitaField> for TraceOneHotColumn {
    fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError> {
        CommitSourceDescriptor::new(
            self.num_vars,
            self.total_field_elems(),
            self.total_field_elems(),
            CommitSourceClass::OneHot {
                chunk_size: self.one_hot_k,
            },
            "jolt-trace-one-hot-batch",
        )
    }

    /// The trace stores hot positions, so every coefficient it commits is
    /// `0` or `1` and no scan is possible or needed.
    fn committed_centered_reach(
        &self,
        _modulus: u128,
        _centering_threshold: u128,
    ) -> Result<(u128, u128), AkitaError> {
        Ok((0, 1))
    }

    fn available_polynomial_types(
        &self,
        _plan: &CommitInnerPlan,
    ) -> Result<AvailablePolynomialTypes, AkitaError> {
        Ok(AvailablePolynomialTypes::external_only())
    }

    fn represent_as(
        &self,
        _selected: PolynomialTypeSelection,
        _plan: &CommitInnerPlan,
    ) -> Result<PolynomialRepresentation<'_, AkitaField>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "trace one-hot sources require their sparse CPU commitment operation".into(),
        ))
    }

    fn external_inner_commitment_capability(
        &self,
        backend: BackendKindId,
        _plan: &CommitInnerPlan,
    ) -> Result<Option<ExternalInnerCommitmentCapability>, AkitaError> {
        let capability = trace_commitment_capability()?;
        Ok((backend == capability.backend()).then_some(capability))
    }

    fn prepare_external_inner_commitment(
        &self,
        selected: ExternalInnerCommitmentCapability,
        _plan: &CommitInnerPlan,
    ) -> Result<PreparedExternalInnerCommitment<'_, AkitaField>, AkitaError> {
        if selected != trace_commitment_capability()? {
            return Err(AkitaError::InvalidInput(
                "trace one-hot source selected a non-CPU commitment operation".into(),
            ));
        }
        PreparedExternalInnerCommitment::new(
            selected,
            self,
            &TraceOneHotColumnCommitOperation,
            None,
        )
    }
}

impl<const D: usize> RootOpeningSource<AkitaField, D> for TraceOneHotColumn {
    type OpeningView<'a>
        = TraceOneHotColumnView<'a, D>
    where
        Self: 'a;
    type OpeningBatchView<'a>
        = TraceOneHotColumnBatchView<'a, D>
    where
        Self: 'a;

    fn opening_view(&self) -> Result<Self::OpeningView<'_>, AkitaError> {
        validate_dimension::<D>(self.one_hot_k)?;
        Ok(TraceOneHotColumnView { source: self })
    }

    fn opening_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::OpeningBatchView<'a>, AkitaError> {
        validate_batch(polys)?;
        validate_dimension::<D>(polys[0].one_hot_k)?;
        Ok(TraceOneHotColumnBatchView { sources: polys })
    }
}

pub(super) fn validate_batch(polys: &[&TraceOneHotColumn]) -> Result<(), AkitaError> {
    let first = polys
        .first()
        .ok_or_else(|| AkitaError::InvalidInput("empty trace batch".into()))?;
    if first.rows.num_rows() != first.num_rows
        || first.rows.num_columns() != first.num_columns
        || polys.len() != first.num_columns
        || polys.iter().enumerate().any(|(column, source)| {
            !Arc::ptr_eq(&source.rows, &first.rows)
                || source.column_index != column
                || source.num_rows != first.num_rows
                || source.num_columns != first.num_columns
                || source.one_hot_k != first.one_hot_k
                || source.num_vars != first.num_vars
        })
    {
        return Err(AkitaError::InvalidInput(
            "trace batch ownership, dimensions, or column order disagree".into(),
        ));
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
