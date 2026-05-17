//! Verifier error types.

#[derive(Debug, thiserror::Error)]
pub enum VerifierError {
    #[error("proof contains mixed clear and ZK stage proofs")]
    InconsistentProofZkMode,

    #[error("verifier functionality has not been implemented yet")]
    Unimplemented,
}
