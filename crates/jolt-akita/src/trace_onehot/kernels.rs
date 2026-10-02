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

use super::commit::commit_columns;
use super::decomposition::{decompose_fold_columns_with_mode, DecomposeRotationMode};
use super::opening::opening_fold_columns;
use super::source::{
    validate_batch, TraceOneHotColumn, TraceOneHotColumnBatchView, TraceOneHotColumnView,
};
use super::traversal::coefficient_packing_partials_columns;
use crate::AkitaField;

pub(super) struct TraceOneHotColumnCommitOperation;

pub(super) fn trace_commitment_capability() -> Result<ExternalInnerCommitmentCapability, AkitaError>
{
    cpu_external_inner_commitment_capability::<
        TraceOneHotColumn,
        TraceOneHotColumnCommitOperation,
        AkitaField,
    >("jolt-trace-one-hot-batch")
}

impl ExternalInnerCommitmentOperation<AkitaField> for TraceOneHotColumnCommitOperation {
    fn identity(&self) -> ExternalOperationIdentity {
        ExternalOperationIdentity::of::<
            TraceOneHotColumn,
            TraceOneHotColumnCommitOperation,
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
        let sources = sources
            .iter()
            .map(|source| source.payload::<TraceOneHotColumn>())
            .collect::<Result<Vec<_>, _>>()?;
        validate_batch(&sources)?;
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            AkitaField,
            plan.ring_dimension,
            |D| commit_columns::<D>(prepared.expanded(), sources[0], *plan)
        )
    }
}

impl<E, const D: usize> OpeningFoldKernel<TraceOneHotColumnView<'_, D>, AkitaField, D>
    for CpuBackend<AkitaField, E>
{
    fn evaluate_and_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: TraceOneHotColumnView<'_, D>,
        plan: OpeningFoldPlan<'_, AkitaField>,
    ) -> Result<OpeningFoldOutput<AkitaField, D>, AkitaError> {
        Ok(opening_fold_columns(source.source(), plan)?.swap_remove(source.source().column_index))
    }

    fn decompose_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        _source: TraceOneHotColumnView<'_, D>,
        _plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness, AkitaError> {
        Err(AkitaError::InvalidInput(
            "trace decomposition requires its native batch".into(),
        ))
    }
}

impl<E, const D: usize> OpeningBatchKernel<TraceOneHotColumnBatchView<'_, D>, AkitaField, D>
    for CpuBackend<AkitaField, E>
{
    fn evaluate_and_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: TraceOneHotColumnBatchView<'_, D>,
        plan: OpeningFoldPlan<'_, AkitaField>,
    ) -> Result<Vec<OpeningFoldOutput<AkitaField, D>>, AkitaError> {
        opening_fold_columns(source.source(), plan)
    }

    fn decompose_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: TraceOneHotColumnBatchView<'_, D>,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<CpuFoldResponses, AkitaError> {
        let source = source.source();
        let (num_positions_per_block, num_digits, _) = plan.scalar_params();
        if num_positions_per_block == 0 {
            return Err(AkitaError::InvalidInput(
                "batched decompose_fold requires positive block geometry".into(),
            ));
        }
        let num_blocks = plan.validate_uniform_batch(std::iter::repeat_n(
            RootPolyShape::<AkitaField, D>::num_live_ring_elems(source)
                .div_ceil(num_positions_per_block),
            source.num_columns,
        ))?;
        let rotation_mode = DecomposeRotationMode::from_env()?;
        match plan {
            DecomposeFoldBatchPlan::Sparse { challenges, .. } => {
                let chunk_ranges = akita_params::dyadic_block_ranges(num_blocks, 1)?;
                let witness = decompose_fold_columns_with_mode::<D>(
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
                let chunks = decompose_fold_columns_with_mode::<D>(
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
    SubringCoefficientPackingBatchKernel<TraceOneHotColumnBatchView<'_, D>, AkitaField, E, D>
    for CpuBackend<AkitaField, E>
where
    E: ExtField<AkitaField> + FpExtEncoding<AkitaField>,
{
    fn coefficient_packing_partials_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: TraceOneHotColumnBatchView<'_, D>,
        plan: SubringCoefficientPackingPlan<'_, E>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<AkitaField>>, AkitaError> {
        coefficient_packing_partials_columns::<E, D>(source.source(), plan)?
            .into_iter()
            .map(|coordinates| {
                SubringCoefficientPackingPartials::new(
                    plan.point.geometry(),
                    plan.point.num_live_blocks(),
                    coordinates,
                )
            })
            .collect()
    }
}
