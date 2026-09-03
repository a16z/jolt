use akita_error::AkitaError;
use akita_prover::compute::{
    CommitInnerPlan, DecomposeFoldBatchPlan, DecomposeFoldPlan, OpeningBatchKernel,
    OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan, RootCommitKernel,
    SubringCoefficientPackingBatchKernel, SubringCoefficientPackingPartials,
    SubringCoefficientPackingPlan,
};
use akita_prover::{
    BatchDecomposeFoldOutcome, CommitInnerWitness, CpuBackend, DecomposeFoldWitness,
};
use akita_types::FpExtEncoding;
use jolt_field::ExtField;
use rayon::prelude::*;

use super::commit::commit_packed;
use super::decomposition::decompose_fold_packed;
use super::opening::opening_fold_packed;
use super::source::{TracePackedOneHotBatchView, TracePackedOneHotView};
use super::traversal::coefficient_packing_partials_packed;
use crate::AkitaField;

impl<const D: usize> RootCommitKernel<TracePackedOneHotView<'_, D>, AkitaField, D> for CpuBackend {
    fn commit_inner_group(
        &self,
        prepared: &Self::PreparedSetup,
        sources: Vec<TracePackedOneHotView<'_, D>>,
        plan: CommitInnerPlan,
    ) -> Result<Vec<CommitInnerWitness<AkitaField>>, AkitaError> {
        sources
            .into_par_iter()
            .map(|source| commit_packed::<D>(self, prepared, source.source(), plan))
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
        opening_fold_packed(source.source(), plan)
    }

    fn decompose_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: TracePackedOneHotView<'_, D>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness<AkitaField>, AkitaError> {
        decompose_fold_packed::<D>(
            source.source(),
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
        let source = source.source();
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
        }
    }
}

impl<E, const D: usize>
    SubringCoefficientPackingBatchKernel<TracePackedOneHotBatchView<'_, D>, AkitaField, E, D>
    for CpuBackend
where
    E: ExtField<AkitaField> + FpExtEncoding<AkitaField> + jolt_field::MulBaseUnreduced<AkitaField>,
{
    fn coefficient_packing_partials_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: TracePackedOneHotBatchView<'_, D>,
        plan: SubringCoefficientPackingPlan<'_, E>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<AkitaField>>, AkitaError> {
        source
            .sources
            .iter()
            .map(|source| {
                let coordinates = coefficient_packing_partials_packed::<E, D>(source, plan)?;
                SubringCoefficientPackingPartials::new(
                    plan.point.geometry(),
                    plan.point.num_live_blocks(),
                    coordinates,
                )
            })
            .collect()
    }
}
