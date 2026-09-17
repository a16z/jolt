use std::any::Any;

use akita_error::AkitaError;
use akita_prover::compute::{
    CommitInnerPlan, DecomposeFoldBatchPlan, DecomposeFoldPlan, OpeningBatchKernel,
    OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan, SubringCoefficientPackingBatchKernel,
    SubringCoefficientPackingPartials, SubringCoefficientPackingPlan,
};
use akita_prover::{
    cpu_external_inner_commitment_capability, cpu_external_inner_prepared_setup,
    BatchDecomposeFoldOutcome, CommitInnerWitness, CpuBackend, CpuPreparedSetup,
    DecomposeFoldWitness, ExternalInnerCommitmentCapability, ExternalInnerCommitmentInput,
    ExternalInnerCommitmentOperation, ExternalOperationIdentity,
};
use akita_types::{dispatch_for_field, FpExtEncoding};
#[expect(
    unused_imports,
    reason = "dispatch_for_field matches these nominal slot tokens without resolving them"
)]
use akita_types::{ProtocolDispatchSlot, RingRole};
use jolt_field::ExtField;
use rayon::prelude::*;

use super::commit::commit_packed;
use super::decomposition::decompose_fold_packed;
use super::opening::opening_fold_packed;
use super::source::{TracePackedOneHot, TracePackedOneHotBatchView, TracePackedOneHotView};
use super::traversal::coefficient_packing_partials_packed;
use crate::AkitaField;

struct TracePackedOneHotCommitAlgorithm;

pub(super) struct TracePackedOneHotCommitOperation;

pub(super) static TRACE_COMMITMENT_OPERATION: TracePackedOneHotCommitOperation =
    TracePackedOneHotCommitOperation;

pub(super) fn trace_commitment_capability() -> Result<ExternalInnerCommitmentCapability, AkitaError>
{
    cpu_external_inner_commitment_capability::<
        TracePackedOneHot,
        TracePackedOneHotCommitAlgorithm,
        AkitaField,
    >("jolt-trace-packed-one-hot")
}

impl ExternalInnerCommitmentOperation<AkitaField> for TracePackedOneHotCommitOperation {
    fn identity(&self) -> ExternalOperationIdentity {
        ExternalOperationIdentity::of::<
            TracePackedOneHot,
            TracePackedOneHotCommitAlgorithm,
            CpuPreparedSetup<AkitaField>,
        >()
    }

    fn commit_group(
        &self,
        plan: &CommitInnerPlan,
        sources: &[ExternalInnerCommitmentInput<'_>],
        context: &dyn Any,
    ) -> Result<Vec<CommitInnerWitness<AkitaField>>, AkitaError> {
        let prepared = cpu_external_inner_prepared_setup::<AkitaField>(context)?;
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            AkitaField,
            plan.ring_dimension,
            |D| sources
                .par_iter()
                .map(|source| {
                    let source = source.payload::<TracePackedOneHot>()?;
                    commit_packed::<D>(&CpuBackend::DEFAULT, prepared, source, *plan)
                })
                .collect()
        )
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
    E: ExtField<AkitaField> + FpExtEncoding<AkitaField>,
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
