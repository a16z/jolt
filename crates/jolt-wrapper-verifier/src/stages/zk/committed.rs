use jolt_crypto::VectorCommitment;
use jolt_sumcheck::CommittedSumcheckProof;

use crate::WrapperError;

use super::outputs::{CommittedOutputClaimOutput, CommittedOutputClaimShape};

pub fn verify_output_claim_commitments<VC>(
    setup: &VC::Setup,
    proof_name: &'static str,
    proof: &CommittedSumcheckProof<VC::Output>,
    output_claim_count: usize,
) -> Result<CommittedOutputClaimOutput<VC::Output>, WrapperError>
where
    VC: VectorCommitment,
{
    let capacity = VC::capacity(setup);
    if capacity == 0 {
        return Err(WrapperError::InvalidVectorCommitmentCapacity {
            required: 1,
            got: 0,
        });
    }

    let expected = output_claim_count.div_ceil(capacity);
    let got = proof.output_claims.commitments.len();
    if got != expected {
        return Err(WrapperError::SpartanSumcheckFailed {
            reason: format!(
                "{proof_name} output-claim commitment count mismatch: expected {expected} for {output_claim_count} hidden claims at VC capacity {capacity}, got {got}"
            ),
        });
    }

    Ok(CommittedOutputClaimOutput {
        shape: CommittedOutputClaimShape {
            output_claim_count,
            row_count: got,
            row_len: capacity,
        },
        commitments: proof.output_claims.clone(),
    })
}
