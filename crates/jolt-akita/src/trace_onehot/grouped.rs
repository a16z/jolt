use std::sync::Arc;

use akita_error::AkitaError;
use akita_prover::backend::{
    DenseBatchView, DenseView, OneHotBatchView, OneHotView, SignedByteBatchView, SignedByteView,
};
use akita_prover::compute::{
    CommitInnerPlan, DecomposeFoldBatchPlan, DecomposeFoldPlan, OpeningBatchKernel,
    OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan, RootCommitKernel,
    SubringCoefficientPackingBatchKernel, SubringCoefficientPackingPartials,
    SubringCoefficientPackingPlan,
};
use akita_prover::{
    BatchDecomposeFoldOutcome, CommitInnerWitness, CpuBackend, DecomposeFoldWitness, DensePoly,
    OneHotPoly, RootCommitSource, RootOpeningSource, RootPolyMeta, RootPolyShape, SignedBytePoly,
};
use akita_types::FpExtEncoding;
use jolt_field::{ExtField, MulBaseUnreduced};

use super::source::{TracePackedOneHot, TracePackedOneHotBatchView, TracePackedOneHotView};
use crate::AkitaField;

/// One polynomial of the heterogeneous grouped opening. Every variant shares
/// the commit-time hint storage instead of borrowing it: akita bounds its
/// opening kernels over every view lifetime (`for<'a>`), which only a
/// `'static` source satisfies.
#[derive(Clone, Debug)]
pub(crate) enum GroupedRootSource {
    Dense(GroupMember<DensePoly<AkitaField>>),
    OneHot(GroupMember<OneHotPoly<AkitaField, u8>>),
    Trace(TracePackedOneHot),
    SignedBytes(GroupMember<SignedBytePoly>),
}

#[derive(Clone, Debug)]
pub(crate) struct GroupMember<T> {
    group: Arc<[T]>,
    index: usize,
}

impl<T> GroupMember<T> {
    pub(crate) fn all(group: &Arc<[T]>) -> impl Iterator<Item = Self> + '_ {
        (0..group.len()).map(|index| Self {
            group: Arc::clone(group),
            index,
        })
    }

    #[expect(
        clippy::indexing_slicing,
        reason = "`all` is the only constructor and yields index < group.len()"
    )]
    pub(super) fn poly(&self) -> &T {
        &self.group[self.index]
    }
}

pub(crate) struct GroupedRootView<'view, const D: usize> {
    pub(super) source: &'view GroupedRootSource,
}

pub(crate) struct GroupedRootBatchView<'view, const D: usize> {
    pub(super) sources: &'view [&'view GroupedRootSource],
}

pub(super) fn mixed_group(operation: &str) -> AkitaError {
    AkitaError::InvalidInput(format!(
        "grouped root {operation} groups must be representation-homogeneous"
    ))
}

impl RootPolyMeta<AkitaField> for GroupedRootSource {
    fn num_vars(&self) -> usize {
        match self {
            Self::Dense(member) => RootPolyMeta::num_vars(member.poly()),
            Self::OneHot(member) => RootPolyMeta::num_vars(member.poly()),
            Self::Trace(poly) => RootPolyMeta::num_vars(poly),
            Self::SignedBytes(member) => RootPolyMeta::<AkitaField>::num_vars(member.poly()),
        }
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        match self {
            Self::Dense(_) | Self::SignedBytes(_) => None,
            Self::OneHot(member) => RootPolyMeta::onehot_chunk_size(member.poly()),
            Self::Trace(poly) => RootPolyMeta::onehot_chunk_size(poly),
        }
    }
}

impl<const D: usize> RootPolyShape<AkitaField, D> for GroupedRootSource {
    fn num_ring_elems(&self) -> usize {
        match self {
            Self::Dense(member) => RootPolyShape::<AkitaField, D>::num_ring_elems(member.poly()),
            Self::OneHot(member) => RootPolyShape::<AkitaField, D>::num_ring_elems(member.poly()),
            Self::Trace(poly) => RootPolyShape::<AkitaField, D>::num_ring_elems(poly),
            Self::SignedBytes(member) => {
                RootPolyShape::<AkitaField, D>::num_ring_elems(member.poly())
            }
        }
    }

    fn num_vars(&self) -> usize {
        match self {
            Self::Dense(member) => RootPolyShape::<AkitaField, D>::num_vars(member.poly()),
            Self::OneHot(member) => RootPolyShape::<AkitaField, D>::num_vars(member.poly()),
            Self::Trace(poly) => RootPolyShape::<AkitaField, D>::num_vars(poly),
            Self::SignedBytes(member) => RootPolyShape::<AkitaField, D>::num_vars(member.poly()),
        }
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        match self {
            Self::Dense(_) | Self::SignedBytes(_) => None,
            Self::OneHot(member) => {
                RootPolyShape::<AkitaField, D>::onehot_chunk_size(member.poly())
            }
            Self::Trace(poly) => RootPolyShape::<AkitaField, D>::onehot_chunk_size(poly),
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
            Self::Dense(member) => RootCommitSource::<AkitaField, D>::committed_centered_reach(
                member.poly(),
                modulus,
                centering_threshold,
            ),
            Self::OneHot(member) => RootCommitSource::<AkitaField, D>::committed_centered_reach(
                member.poly(),
                modulus,
                centering_threshold,
            ),
            Self::Trace(poly) => RootCommitSource::<AkitaField, D>::committed_centered_reach(
                poly,
                modulus,
                centering_threshold,
            ),
            Self::SignedBytes(member) => {
                RootCommitSource::<AkitaField, D>::committed_centered_reach(
                    member.poly(),
                    modulus,
                    centering_threshold,
                )
            }
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
                    .map(|view| match view.source {
                        GroupedRootSource::Dense(member) => member.poly().commit_view(),
                        GroupedRootSource::OneHot(_)
                        | GroupedRootSource::Trace(_)
                        | GroupedRootSource::SignedBytes(_) => Err(mixed_group("commitment")),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                RootCommitKernel::<DenseView<'_, AkitaField, D>, AkitaField, D>::commit_inner_group(
                    self, prepared, dense, plan,
                )
            }
            GroupedRootSource::OneHot(_) => {
                let one_hot = sources
                    .into_iter()
                    .map(|view| match view.source {
                        GroupedRootSource::OneHot(member) => member.poly().commit_view(),
                        GroupedRootSource::Dense(_)
                        | GroupedRootSource::Trace(_)
                        | GroupedRootSource::SignedBytes(_) => Err(mixed_group("commitment")),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                RootCommitKernel::<OneHotView<'_, AkitaField, D, u8>, AkitaField, D>::commit_inner_group(
                    self, prepared, one_hot, plan,
                )
            }
            GroupedRootSource::Trace(_) => {
                let trace = sources
                    .into_iter()
                    .map(|view| match view.source {
                        GroupedRootSource::Trace(poly) => poly.commit_view(),
                        GroupedRootSource::Dense(_)
                        | GroupedRootSource::OneHot(_)
                        | GroupedRootSource::SignedBytes(_) => Err(mixed_group("commitment")),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                RootCommitKernel::<TracePackedOneHotView<'_, D>, AkitaField, D>::commit_inner_group(
                    self, prepared, trace, plan,
                )
            }
            GroupedRootSource::SignedBytes(_) => {
                let bytes = sources
                    .into_iter()
                    .map(|view| match view.source {
                        GroupedRootSource::SignedBytes(member) => {
                            RootCommitSource::<AkitaField, D>::commit_view(member.poly())
                        }
                        GroupedRootSource::Dense(_)
                        | GroupedRootSource::OneHot(_)
                        | GroupedRootSource::Trace(_) => Err(mixed_group("commitment")),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                RootCommitKernel::<SignedByteView<'_, D>, AkitaField, D>::commit_inner_group(
                    self, prepared, bytes, plan,
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
            GroupedRootSource::Dense(member) => {
                OpeningFoldKernel::<DenseView<'_, AkitaField, D>, AkitaField, D>::evaluate_and_fold(
                    self,
                    prepared,
                    member.poly().opening_view()?,
                    plan,
                )
            }
            GroupedRootSource::OneHot(member) => OpeningFoldKernel::<
                OneHotView<'_, AkitaField, D, u8>,
                AkitaField,
                D,
            >::evaluate_and_fold(
                self,
                prepared,
                member.poly().opening_view()?,
                plan,
            ),
            GroupedRootSource::Trace(poly) => OpeningFoldKernel::<
                TracePackedOneHotView<'_, D>,
                AkitaField,
                D,
            >::evaluate_and_fold(
                self, prepared, poly.opening_view()?, plan
            ),
            GroupedRootSource::SignedBytes(member) => {
                OpeningFoldKernel::<SignedByteView<'_, D>, AkitaField, D>::evaluate_and_fold(
                    self,
                    prepared,
                    RootOpeningSource::<AkitaField, D>::opening_view(member.poly())?,
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
            GroupedRootSource::Dense(member) => {
                OpeningFoldKernel::<DenseView<'_, AkitaField, D>, AkitaField, D>::decompose_fold(
                    self,
                    prepared,
                    member.poly().opening_view()?,
                    plan,
                )
            }
            GroupedRootSource::OneHot(member) => OpeningFoldKernel::<
                OneHotView<'_, AkitaField, D, u8>,
                AkitaField,
                D,
            >::decompose_fold(
                self,
                prepared,
                member.poly().opening_view()?,
                plan,
            ),
            GroupedRootSource::Trace(poly) => OpeningFoldKernel::<
                TracePackedOneHotView<'_, D>,
                AkitaField,
                D,
            >::decompose_fold(
                self, prepared, poly.opening_view()?, plan
            ),
            GroupedRootSource::SignedBytes(member) => {
                OpeningFoldKernel::<SignedByteView<'_, D>, AkitaField, D>::decompose_fold(
                    self,
                    prepared,
                    RootOpeningSource::<AkitaField, D>::opening_view(member.poly())?,
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
                        GroupedRootSource::Dense(member) => Ok(member.poly()),
                        GroupedRootSource::OneHot(_)
                        | GroupedRootSource::Trace(_)
                        | GroupedRootSource::SignedBytes(_) => Err(mixed_group("opening")),
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
                        GroupedRootSource::OneHot(member) => Ok(member.poly()),
                        GroupedRootSource::Dense(_)
                        | GroupedRootSource::Trace(_)
                        | GroupedRootSource::SignedBytes(_) => Err(mixed_group("opening")),
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
                        GroupedRootSource::Trace(poly) => Ok(poly),
                        GroupedRootSource::Dense(_)
                        | GroupedRootSource::OneHot(_)
                        | GroupedRootSource::SignedBytes(_) => Err(mixed_group("opening")),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let view =
                    <TracePackedOneHot as RootOpeningSource<AkitaField, D>>::opening_batch(&trace)?;
                OpeningBatchKernel::<TracePackedOneHotBatchView<'_, D>, AkitaField, D>::decompose_fold_batch(
                    self, prepared, view, plan,
                )
            }
            GroupedRootSource::SignedBytes(_) => {
                let bytes = source
                    .sources
                    .iter()
                    .map(|source| match source {
                        GroupedRootSource::SignedBytes(member) => Ok(member.poly()),
                        GroupedRootSource::Dense(_)
                        | GroupedRootSource::OneHot(_)
                        | GroupedRootSource::Trace(_) => Err(mixed_group("opening")),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let view =
                    <SignedBytePoly as RootOpeningSource<AkitaField, D>>::opening_batch(&bytes)?;
                OpeningBatchKernel::<SignedByteBatchView<'_, D>, AkitaField, D>::decompose_fold_batch(
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
                        GroupedRootSource::Dense(member) => Ok(member.poly()),
                        GroupedRootSource::OneHot(_)
                        | GroupedRootSource::Trace(_)
                        | GroupedRootSource::SignedBytes(_) => {
                            Err(mixed_group("coefficient-packing"))
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
                        GroupedRootSource::OneHot(member) => Ok(member.poly()),
                        GroupedRootSource::Dense(_)
                        | GroupedRootSource::Trace(_)
                        | GroupedRootSource::SignedBytes(_) => {
                            Err(mixed_group("coefficient-packing"))
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
                        GroupedRootSource::Trace(poly) => Ok(poly),
                        GroupedRootSource::Dense(_)
                        | GroupedRootSource::OneHot(_)
                        | GroupedRootSource::SignedBytes(_) => {
                            Err(mixed_group("coefficient-packing"))
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
            GroupedRootSource::SignedBytes(_) => {
                let bytes = source
                    .sources
                    .iter()
                    .map(|source| match source {
                        GroupedRootSource::SignedBytes(member) => Ok(member.poly()),
                        GroupedRootSource::Dense(_)
                        | GroupedRootSource::OneHot(_)
                        | GroupedRootSource::Trace(_) => Err(mixed_group("coefficient-packing")),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let view =
                    <SignedBytePoly as RootOpeningSource<AkitaField, D>>::opening_batch(&bytes)?;
                SubringCoefficientPackingBatchKernel::<
                    SignedByteBatchView<'_, D>,
                    AkitaField,
                    E,
                    D,
                >::coefficient_packing_partials_batch(self, prepared, view, plan)
            }
        }
    }
}
