//! Native CPU implementation of Akita's trace commitment kernel.

use std::sync::Arc;

use jolt_akita::{TraceOneHotCommitment, TraceOneHotRows};
use jolt_field::JoltField;
use jolt_openings::{CommitmentScheme, GroupSetupMetadata, TransparentObjectSetup};
use jolt_witness::JoltWitnessPlane;

use crate::akita::commitment::{CommitWitness, WitnessCommitRequest, WitnessCommitment};
use crate::{KernelError, ProofSession, ReferenceBackend};

use super::witness::assemble_one_hot_trace_rows;

impl<F, PCS> CommitWitness<F, PCS> for ReferenceBackend
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + TraceOneHotCommitment + TransparentObjectSetup,
    PCS::ProverSetup: GroupSetupMetadata,
{
    fn commit_witness(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        request: WitnessCommitRequest<'_, PCS>,
    ) -> Result<WitnessCommitment<PCS>, KernelError<F>> {
        let assembled = assemble_one_hot_trace_rows(
            witness,
            request.plan(),
            request.shape().ra_layout,
            request.shape().log_k_chunk,
            request.shape().log_t,
        )?;
        #[cfg(feature = "field-inline")]
        let field_inc = super::field_inline::commit_field_inc::<F, PCS>(
            request.setup(),
            request.shape().log_t,
            assembled.increments.clone(),
        )?;
        let fused_inc = super::increments::commit_fused_inc::<F, PCS>(
            request.setup(),
            request.shape().log_t,
            witness,
        )?;
        let mut group_hints = request.precommitted_hints().to_vec();
        #[cfg(feature = "field-inline")]
        group_hints.push(&field_inc.hint);
        group_hints.push(&fused_inc.hint);
        let precommitted_hints = group_hints.as_slice();
        let committed = PCS::commit_trace_one_hot(
            request.setup(),
            request.plan().layout_digest(),
            Arc::clone(&assembled.rows) as Arc<dyn TraceOneHotRows>,
            precommitted_hints,
        );
        assembled.rows.check_extraction()?;
        let (commitment, hint) = committed?;
        PCS::release_post_commit_residency(request.setup())?;
        #[cfg(feature = "field-inline")]
        _session.park(assembled.increments);
        Ok(WitnessCommitment {
            fused_inc,
            commitment,
            hint,
            #[cfg(feature = "field-inline")]
            field_inc,
        })
    }
}
