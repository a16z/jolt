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

    #[error("variable-point KZG batch has inconsistent dimensions")]
    InvalidBatchShape,

    #[error("variable-point KZG batch requires three distinct points per polynomial")]
    RepeatedBatchPoint,

    #[error("variable-point KZG quotient division left a nonzero remainder")]
    NonzeroQuotientRemainder,

    #[error("degree-bounded KZG supports degree 5, got {0}")]
    UnsupportedDegreeBound(usize),

    #[error("KZG degree-bound pairing check failed")]
    DegreeBoundCheckFailed,

    #[error("variable-point KZG pairing check failed")]
    VariableBatchCheckFailed,
}
