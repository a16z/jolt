//! PCS error types.

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum OpeningsError {
    #[error("opening proof verification failed")]
    VerificationFailed,

    #[error("commitment mismatch: expected {expected}, got {actual}")]
    CommitmentMismatch { expected: String, actual: String },

    #[error("invalid setup parameters: {0}")]
    InvalidSetup(String),

    #[error("invalid batch opening: {0}")]
    InvalidBatch(String),

    #[error("commitment failed: {0}")]
    CommitFailed(String),

    #[error("opening proof generation failed: {0}")]
    ProveFailed(String),

    #[error("polynomial size {poly_size} exceeds setup max {setup_max}")]
    PolynomialTooLarge { poly_size: usize, setup_max: usize },
}
