//! Shared checks for committed sumcheck stage boundaries.

use common::constants::MAX_BLINDFOLD_GENERATORS;
use jolt_claims::protocols::jolt::JoltStageId;
use jolt_field::Field;
use jolt_sumcheck::SumcheckProof;

use crate::{verifier::CheckedInputs, VerifierError};

pub(crate) fn zk_vector_commitment_capacity_requirement() -> usize {
    MAX_BLINDFOLD_GENERATORS
}

pub(crate) fn require_output_claim_commitments<F, C>(
    checked: &CheckedInputs,
    proof: &SumcheckProof<F, C>,
    proof_name: &'static str,
    output_claim_scalars: usize,
    stage: JoltStageId,
) -> Result<(), VerifierError>
where
    F: Field,
{
    let capacity = checked
        .vc_capacity
        .ok_or(VerifierError::MissingVectorCommitmentSetup)?;
    let required = zk_vector_commitment_capacity_requirement();
    if capacity < required {
        return Err(VerifierError::InvalidVectorCommitmentCapacity {
            required,
            got: capacity,
        });
    }
    let expected = output_claim_scalars.div_ceil(capacity);
    let committed = proof
        .as_committed()
        .ok_or(VerifierError::ExpectedCommittedProof { field: proof_name })?;
    let got = committed.output_claims.commitments.len();
    if got != expected {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage,
            reason: format!(
                "{proof_name} output-claim commitment count mismatch: expected {expected} for {output_claim_scalars} hidden claims at VC capacity {capacity}, got {got}"
            ),
        });
    }

    Ok(())
}
