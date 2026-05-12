//! Error types for HyperKZG operations.

/// Errors produced by the HyperKZG commitment scheme.
#[derive(Debug, thiserror::Error)]
pub enum HyperKZGError {
    #[error("SRS too small: have {have} powers, need {need}")]
    SrsTooSmall { have: usize, need: usize },

    #[error("expected {expected} intermediate commitments, got {got}")]
    WrongCommitmentCount { expected: usize, got: usize },

    #[error("each evaluation row must have {expected} entries")]
    WrongEvaluationWidth { expected: usize },

    #[error("polynomial must have at least 1 variable")]
    EmptyPoint,

    #[error("folding consistency check failed at level {level}")]
    FoldingConsistencyFailed { level: usize },

    #[error("batch KZG pairing check failed")]
    PairingCheckFailed,

    #[error("degenerate Fiat-Shamir challenge: r = 0")]
    DegenerateChallenge,
}
