//! Metal routes for the prefix-packed trace one-hot polynomial (ported from the
//! 2026-09-01 port line; compiles only against the fork's Metal-enabled Akita).

use akita_error::AkitaError;
use akita_prover::backend::{DenseBatchView, DenseView, OneHotBatchView, OneHotView};
use akita_prover::compute::{
    CommitInnerPlan, DecomposeFoldBatchPlan, DecomposeFoldPlan, OpeningBatchKernel,
    OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan, RootCommitKernel,
    SubringCoefficientPackingBatchKernel, SubringCoefficientPackingPartials,
    SubringCoefficientPackingPlan,
};
use akita_prover::{
    BatchDecomposeFoldOutcome, CommitInnerWitness, CpuBackend, DecomposeFoldWitness, DensePoly,
    OneHotPoly, RootOpeningSource,
};

use super::commit::commit_packed;
use super::grouped::{grouped_singleton, GroupedRootBatchView, GroupedRootSource, GroupedRootView};
use super::source::{TracePackedOneHot, TracePackedOneHotBatchView, TracePackedOneHotView};
use crate::AkitaField;

#[cfg(all(feature = "metal", target_os = "macos"))]
fn packed_metal_view(
    source: &TracePackedOneHot,
) -> Result<akita_metal::PackedOneHotCommitView<'_>, AkitaError> {
    let selectors = source.rows.packed_selectors().ok_or_else(|| {
        AkitaError::InvalidInput("Metal trace opening requires resident packed selectors".into())
    })?;
    match (
        source.one_hot_k,
        selectors.hot_entries(),
        selectors.zero_suffix_start(),
    ) {
        (256, Some(hot_entries), Some(zero_suffix_start)) => {
            akita_metal::PackedOneHotCommitView::new_k256_with_precomputed_metrics(
                source.column_capacity,
                source.num_columns,
                selectors.row_major(),
                selectors.active_zero_rows(),
                selectors.zero_column_mask(),
                hot_entries,
                zero_suffix_start,
            )
        }
        (256, Some(hot_entries), None) => {
            akita_metal::PackedOneHotCommitView::new_k256_with_precomputed_hot_entries(
                source.column_capacity,
                source.num_columns,
                selectors.row_major(),
                selectors.active_zero_rows(),
                selectors.zero_column_mask(),
                hot_entries,
            )
        }
        _ => akita_metal::PackedOneHotCommitView::new_with_active_zero_rows(
            source.one_hot_k,
            source.column_capacity,
            source.num_columns,
            selectors.row_major(),
            selectors.active_zero_rows(),
            selectors.zero_column_mask(),
        ),
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl<const D: usize> RootCommitKernel<TracePackedOneHotView<'_, D>, AkitaField, D>
    for akita_metal::MetalBackend
{
    fn commit_inner_group(
        &self,
        prepared: &Self::PreparedSetup,
        sources: Vec<TracePackedOneHotView<'_, D>>,
        plan: CommitInnerPlan,
    ) -> Result<Vec<CommitInnerWitness<AkitaField>>, AkitaError> {
        let [view] = sources.as_slice() else {
            return Err(AkitaError::InvalidInput(format!(
                "Metal trace commitment requires one physical source, got {}",
                sources.len()
            )));
        };
        let source = view.source();
        if source.rows.packed_selectors().is_none() {
            let cpu = self.cpu_backend();
            return commit_packed::<D>(&cpu, prepared.cpu_prepared(), source, plan)
                .map(|witness| vec![witness]);
        }
        if D != 512 {
            let cpu = self.cpu_backend();
            return commit_packed::<D>(&cpu, prepared.cpu_prepared(), source, plan)
                .map(|witness| vec![witness]);
        }
        let packed = packed_metal_view(source)?;
        self.commit_packed_onehot::<D>(prepared, packed, plan)
            .map(|witness| vec![witness])
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl<const D: usize> OpeningFoldKernel<TracePackedOneHotView<'_, D>, AkitaField, D>
    for akita_metal::MetalBackend
{
    fn evaluate_and_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: TracePackedOneHotView<'_, D>,
        plan: OpeningFoldPlan<'_, AkitaField>,
    ) -> Result<OpeningFoldOutput<AkitaField, D>, AkitaError> {
        self.record_opening_cpu_fallback(
            source
                .source()
                .num_rows
                .saturating_mul(source.source().num_columns),
        )
        .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
        CpuBackend::DEFAULT.evaluate_and_fold(None, source, plan)
    }

    #[tracing::instrument(skip_all, name = "TracePackedOneHot::decompose_fold_metal_single")]
    fn decompose_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: TracePackedOneHotView<'_, D>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness<AkitaField>, AkitaError> {
        if D != 512 || source.source().one_hot_k != 256 || source.source().column_capacity != 32 {
            self.record_opening_cpu_fallback(1)
                .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
            return CpuBackend::DEFAULT.decompose_fold(None, source, plan);
        }
        self.decompose_fold_packed_onehot::<D>(packed_metal_view(source.source())?, plan)
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl<const D: usize> OpeningBatchKernel<TracePackedOneHotBatchView<'_, D>, AkitaField, D>
    for akita_metal::MetalBackend
{
    #[tracing::instrument(skip_all, name = "TracePackedOneHot::decompose_fold_metal_batch")]
    fn decompose_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: TracePackedOneHotBatchView<'_, D>,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<BatchDecomposeFoldOutcome<AkitaField, D>, AkitaError> {
        if D != 512 || source.source().one_hot_k != 256 || source.source().column_capacity != 32 {
            self.record_opening_cpu_fallback(1)
                .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
            return CpuBackend::DEFAULT.decompose_fold_batch(None, source, plan);
        }
        let DecomposeFoldBatchPlan::Sparse {
            challenges,
            num_positions_per_block,
            num_digits,
            log_basis,
        } = plan;
        Ok(BatchDecomposeFoldOutcome::Fused(
            self.decompose_fold_packed_onehot::<D>(
                packed_metal_view(source.source())?,
                DecomposeFoldPlan {
                    challenges,
                    num_positions_per_block,
                    num_digits,
                    log_basis,
                },
            )?,
        ))
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl<const D: usize>
    SubringCoefficientPackingBatchKernel<
        TracePackedOneHotBatchView<'_, D>,
        AkitaField,
        AkitaField,
        D,
    > for akita_metal::MetalBackend
{
    #[tracing::instrument(skip_all, name = "TracePackedOneHot::coefficient_packing_metal")]
    fn coefficient_packing_partials_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: TracePackedOneHotBatchView<'_, D>,
        plan: SubringCoefficientPackingPlan<'_, AkitaField>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<AkitaField>>, AkitaError> {
        let trace = source.source();
        let packed = packed_metal_view(trace)?;
        self.packed_onehot_coefficient_packing::<D>(packed, plan.point)
            .map(|partials| vec![partials])
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl<const D: usize> OpeningFoldKernel<GroupedRootView<'_, D>, AkitaField, D>
    for akita_metal::MetalBackend
{
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
    ) -> Result<DecomposeFoldWitness<AkitaField>, AkitaError> {
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

#[cfg(all(feature = "metal", target_os = "macos"))]
impl<const D: usize> OpeningBatchKernel<GroupedRootBatchView<'_, D>, AkitaField, D>
    for akita_metal::MetalBackend
{
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
                                    .into(),
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
                                    .into(),
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
                                    .into(),
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

#[cfg(all(feature = "metal", target_os = "macos"))]
impl<const D: usize>
    SubringCoefficientPackingBatchKernel<GroupedRootBatchView<'_, D>, AkitaField, AkitaField, D>
    for akita_metal::MetalBackend
{
    fn coefficient_packing_partials_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: GroupedRootBatchView<'_, D>,
        plan: SubringCoefficientPackingPlan<'_, AkitaField>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<AkitaField>>, AkitaError> {
        if matches!(source.sources.first(), Some(GroupedRootSource::Trace(_))) {
            let trace = source
                .sources
                .iter()
                .map(|source| match source {
                    GroupedRootSource::Trace(polys) => Ok(grouped_singleton(polys)),
                    GroupedRootSource::Dense(_) | GroupedRootSource::OneHot(_) => {
                        Err(AkitaError::InvalidInput(
                            "grouped root coefficient-packing groups must be representation-homogeneous"
                                .to_string(),
                        ))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            let view =
                <TracePackedOneHot as RootOpeningSource<AkitaField, D>>::opening_batch(&trace)?;
            return SubringCoefficientPackingBatchKernel::<
                TracePackedOneHotBatchView<'_, D>,
                AkitaField,
                AkitaField,
                D,
            >::coefficient_packing_partials_batch(self, None, view, plan);
        }

        self.record_opening_cpu_fallback(1)
            .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
        CpuBackend::DEFAULT.coefficient_packing_partials_batch(None, source, plan)
    }
}
