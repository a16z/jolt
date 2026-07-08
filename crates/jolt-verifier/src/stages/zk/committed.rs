//! Shared checks for committed sumcheck stage boundaries.

use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_field::Field;
use jolt_sumcheck::SumcheckProof;

use crate::{verifier::CheckedInputs, VerifierError};

pub(crate) use crate::stages::zk::outputs::CommittedOutputClaimOutput;

pub(crate) fn verify_output_claim_commitments<F, C>(
    checked: &CheckedInputs,
    proof: &SumcheckProof<F, C>,
    proof_label: &'static str,
    output_claim_count: usize,
    stage: JoltRelationId,
) -> Result<CommittedOutputClaimOutput<C>, VerifierError>
where
    F: Field,
    C: Clone,
{
    // Invariant: Some(capacity) implies capacity >= MAX_BLINDFOLD_GENERATORS,
    // enforced by validate_zk_vector_commitment_setup before any stage runs (also
    // guards the div_ceil below against zero).
    let capacity = checked
        .vc_capacity
        .ok_or(VerifierError::MissingVectorCommitmentSetup)?;
    let expected = output_claim_count.div_ceil(capacity);
    let committed = proof
        .as_committed()
        .ok_or(VerifierError::ExpectedCommittedProof { field: proof_label })?;
    let got = committed.output_claims.commitments.len();
    if got != expected {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage,
            reason: format!(
                "{proof_label} output-claim commitment count mismatch: expected {expected} for {output_claim_count} hidden claims at VC capacity {capacity}, got {got}",
            ),
        });
    }

    Ok(CommittedOutputClaimOutput {
        shape: super::outputs::CommittedOutputClaimShape {
            output_claim_count,
            row_len: capacity,
        },
        commitments: committed.output_claims.clone(),
    })
}
