//! Shared checks for committed sumcheck stage boundaries.

use crate::VerifierError;
use common::constants::MAX_BLINDFOLD_GENERATORS;

pub(crate) use crate::stages::zk::inputs::CommittedOutputClaimInputs;
pub(crate) use crate::stages::zk::outputs::CommittedOutputClaimOutput;

pub fn zk_vector_commitment_capacity_requirement() -> usize {
    MAX_BLINDFOLD_GENERATORS
}

pub(crate) fn verify_output_claim_commitments<C>(
    input: CommittedOutputClaimInputs<'_, C>,
) -> Result<CommittedOutputClaimOutput<C>, VerifierError>
where
    C: Clone,
{
    let capacity = input
        .checked
        .vc_capacity
        .ok_or(VerifierError::MissingVectorCommitmentSetup)?;
    let required = zk_vector_commitment_capacity_requirement();
    if capacity < required {
        return Err(VerifierError::InvalidVectorCommitmentCapacity {
            required,
            got: capacity,
        });
    }
    let expected = input.output_claim_count.div_ceil(capacity);
    let got = input.output_claims.commitments.len();
    if got != expected {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: input.stage,
            reason: format!(
                "{} output-claim commitment count mismatch: expected {expected} for {} hidden claims at VC capacity {capacity}, got {got}",
                input.proof_label,
                input.output_claim_count,
            ),
        });
    }

    Ok(CommittedOutputClaimOutput {
        shape: super::outputs::CommittedOutputClaimShape {
            output_claim_count: input.output_claim_count,
            row_count: got,
            row_len: capacity,
        },
        commitments: input.output_claims.clone(),
    })
}
