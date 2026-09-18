use std::sync::Arc;

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
    OneHotPoly, RootCommitSource, RootOpeningSource, RootPolyMeta, RootPolyShape,
};
use akita_types::FpExtEncoding;
use jolt_field::{ExtField, MulBaseUnreduced};

use super::source::{TracePackedOneHot, TracePackedOneHotBatchView, TracePackedOneHotView};
use crate::AkitaField;

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

impl<E, const D: usize>
    SubringCoefficientPackingBatchKernel<GroupedRootBatchView<'_, D>, AkitaField, E, D>
    for CpuBackend
where
    E: ExtField<AkitaField> + FpExtEncoding<AkitaField> + MulBaseUnreduced<AkitaField>,
{
    fn coefficient_packing_partials_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: GroupedRootBatchView<'_, D>,
        plan: SubringCoefficientPackingPlan<'_, E>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<AkitaField>>, AkitaError> {
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
                                "grouped root coefficient-packing groups must be representation-homogeneous"
                                    .to_string(),
                            ))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let view =
                    <DensePoly<AkitaField> as RootOpeningSource<AkitaField, D>>::opening_batch(
                        &dense,
                    )?;
                SubringCoefficientPackingBatchKernel::<
                    DenseBatchView<'_, AkitaField, D>,
                    AkitaField,
                    E,
                    D,
                >::coefficient_packing_partials_batch(self, prepared, view, plan)
            }
            GroupedRootSource::OneHot(_) => {
                let one_hot = source
                    .sources
                    .iter()
                    .map(|source| match source {
                        GroupedRootSource::OneHot(polys) => Ok(grouped_singleton(polys)),
                        GroupedRootSource::Dense(_) | GroupedRootSource::Trace(_) => {
                            Err(AkitaError::InvalidInput(
                                "grouped root coefficient-packing groups must be representation-homogeneous"
                                    .to_string(),
                            ))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let view = <OneHotPoly<AkitaField, u8> as RootOpeningSource<
                    AkitaField,
                    D,
                >>::opening_batch(&one_hot)?;
                SubringCoefficientPackingBatchKernel::<
                    OneHotBatchView<'_, AkitaField, D, u8>,
                    AkitaField,
                    E,
                    D,
                >::coefficient_packing_partials_batch(self, prepared, view, plan)
            }
            GroupedRootSource::Trace(_) => {
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
                SubringCoefficientPackingBatchKernel::<
                    TracePackedOneHotBatchView<'_, D>,
                    AkitaField,
                    E,
                    D,
                >::coefficient_packing_partials_batch(self, prepared, view, plan)
            }
        }
    }
}
