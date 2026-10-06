use std::any::Any;

use akita_error::AkitaError;
use akita_params::dispatch_for_field;
#[expect(
    unused_imports,
    reason = "dispatch_for_field matches these nominal slot tokens without resolving them"
)]
use akita_params::{ProtocolDispatchSlot, RingRole};
use akita_pcs::custom_source::{
    cpu_external_inner_commitment_capability, cpu_external_inner_prepared_setup, CommitInnerPlan,
    CpuFoldResponses, CpuPreparedSetup, DecomposeFoldBatchPlan, DecomposeFoldPlan,
    DecomposeFoldWitness, ExternalInnerCommitmentCapability, ExternalInnerCommitmentInput,
    ExternalInnerCommitmentOperation, ExternalOperationIdentity, OpeningBatchKernel,
    OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan, RootPolyShape,
    SubringCoefficientPackingBatchKernel, SubringCoefficientPackingPartials,
    SubringCoefficientPackingPlan,
};
use akita_pcs::CpuBackend;
use akita_types::{FpExtEncoding, RingVec};
use jolt_field::ExtField;
use rayon::prelude::*;

use super::commit::commit_packed;
use super::decomposition::{decompose_fold_packed_with_mode, DecomposeRotationMode};
use super::opening::opening_fold_packed;
use super::source::{TracePackedOneHot, TracePackedOneHotBatchView, TracePackedOneHotView};
use super::traversal::coefficient_packing_partials_packed;
use crate::AkitaField;

pub(super) struct TracePackedOneHotCommitOperation;

pub(super) fn trace_commitment_capability() -> Result<ExternalInnerCommitmentCapability, AkitaError>
{
    cpu_external_inner_commitment_capability::<
        TracePackedOneHot,
        TracePackedOneHotCommitOperation,
        AkitaField,
    >("jolt-trace-packed-one-hot")
}

impl ExternalInnerCommitmentOperation<AkitaField> for TracePackedOneHotCommitOperation {
    fn identity(&self) -> ExternalOperationIdentity {
        ExternalOperationIdentity::of::<
            TracePackedOneHot,
            TracePackedOneHotCommitOperation,
            CpuPreparedSetup<AkitaField>,
        >()
    }

    fn commit_group(
        &self,
        plan: &CommitInnerPlan,
        sources: &[ExternalInnerCommitmentInput<'_>],
        context: &dyn Any,
    ) -> Result<Vec<RingVec<AkitaField>>, AkitaError> {
        let prepared = cpu_external_inner_prepared_setup::<AkitaField>(context)?;
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            AkitaField,
            plan.ring_dimension,
            |D| sources
                .par_iter()
                .map(|source| {
                    let source = source.payload::<TracePackedOneHot>()?;
                    commit_packed::<D>(prepared.expanded(), source, *plan)
                })
                .collect()
        )
    }
}

impl<E, const D: usize> OpeningFoldKernel<TracePackedOneHotView<'_, D>, AkitaField, D>
    for CpuBackend<AkitaField, E>
{
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
    ) -> Result<DecomposeFoldWitness, AkitaError> {
        let chunk_ranges = akita_params::dyadic_block_ranges(plan.challenges.len(), 1)?;
        decompose_fold_packed_with_mode::<D>(
            source.source(),
            plan.challenges,
            &chunk_ranges,
            plan.num_positions_per_block,
            plan.num_digits,
            DecomposeRotationMode::from_env()?,
        )?
        .into_iter()
        .next()
        .ok_or_else(|| AkitaError::InvalidInput("decompose fold returned no witness".to_string()))
    }
}

impl<E, const D: usize> OpeningBatchKernel<TracePackedOneHotBatchView<'_, D>, AkitaField, D>
    for CpuBackend<AkitaField, E>
{
    fn decompose_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: TracePackedOneHotBatchView<'_, D>,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<CpuFoldResponses, AkitaError> {
        let source = source.source();
        let (num_positions_per_block, num_digits, _) = plan.scalar_params();
        if num_positions_per_block == 0 {
            return Err(AkitaError::InvalidInput(
                "batched decompose_fold requires positive block geometry".into(),
            ));
        }
        let num_blocks = plan.validate_uniform_batch(std::iter::once(
            RootPolyShape::<AkitaField, D>::num_live_ring_elems(source)
                .div_ceil(num_positions_per_block),
        ))?;
        let rotation_mode = DecomposeRotationMode::from_env()?;
        match plan {
            DecomposeFoldBatchPlan::Sparse { challenges, .. } => {
                let chunk_ranges = akita_params::dyadic_block_ranges(num_blocks, 1)?;
                let witness = decompose_fold_packed_with_mode::<D>(
                    source,
                    challenges,
                    &chunk_ranges,
                    num_positions_per_block,
                    num_digits,
                    rotation_mode,
                )?
                .into_iter()
                .next()
                .ok_or_else(|| {
                    AkitaError::InvalidInput("decompose fold returned no witness".into())
                })?;
                Ok(CpuFoldResponses::sparse(witness))
            }
            DecomposeFoldBatchPlan::SparseChunked {
                challenges,
                chunk_ranges,
                ..
            } => {
                let chunks = decompose_fold_packed_with_mode::<D>(
                    source,
                    challenges.as_slice(),
                    chunk_ranges,
                    num_positions_per_block,
                    num_digits,
                    rotation_mode,
                )?;
                CpuFoldResponses::chunked::<D>(chunks)
            }
        }
    }
}

impl<E, const D: usize>
    SubringCoefficientPackingBatchKernel<TracePackedOneHotBatchView<'_, D>, AkitaField, E, D>
    for CpuBackend<AkitaField, E>
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
