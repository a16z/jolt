//! Verifier error types.

#[derive(Debug, thiserror::Error)]
pub enum VerifierError {
    #[error("proof field {field} must be clear for non-ZK verification")]
    ExpectedClearProof { field: &'static str },

    #[error("proof field {field} must be committed for ZK verification")]
    ExpectedCommittedProof { field: &'static str },

    #[error("clear proof is missing opening claims")]
    MissingOpeningClaims,

    #[error("clear proof unexpectedly includes a BlindFold proof")]
    UnexpectedBlindFoldProof,

    #[error("committed proof is missing a BlindFold proof")]
    MissingBlindFoldProof,

    #[error("committed proof unexpectedly includes opening claims")]
    UnexpectedOpeningClaims,

    #[error("verifier functionality has not been implemented yet")]
    Unimplemented,
}
