//! Shared checks for committed sumcheck stage boundaries.

use crate::{verifier::CheckedInputs, VerifierError};

pub(crate) use crate::stages::zk::outputs::{
    CommittedOutputClaimOutput, CommittedOutputClaimShape,
};

/// The row layout of `output_claim_count` hidden claims at the build's vector
/// commitment capacity, which fixes how many output-claim commitments a
/// committed sumcheck sends.
pub(crate) fn output_claim_shape(
    checked: &CheckedInputs,
    output_claim_count: usize,
) -> Result<CommittedOutputClaimShape, VerifierError> {
    // Invariant: Some(capacity) implies capacity >= MAX_BLINDFOLD_GENERATORS,
    // enforced by validate_zk_vector_commitment_setup before any stage runs (also
    // guards the div_ceil in `row_count` against zero).
    let capacity = checked
        .vc_capacity
        .ok_or(VerifierError::MissingVectorCommitmentSetup)?;
    Ok(CommittedOutputClaimShape {
        output_claim_count,
        row_len: capacity,
    })
}
