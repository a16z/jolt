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
        #[cfg(feature = "field-inline")]
        let group_hints = request
            .precommitted_hints()
            .iter()
            .copied()
            .chain(std::iter::once(&field_inc.hint))
            .collect::<Vec<_>>();
        #[cfg(feature = "field-inline")]
        let precommitted_hints = group_hints.as_slice();
        #[cfg(not(feature = "field-inline"))]
        let precommitted_hints = request.precommitted_hints();
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
            commitment,
            hint,
            #[cfg(feature = "field-inline")]
            field_inc,
        })
    }
}
