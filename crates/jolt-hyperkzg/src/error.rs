//! Error types for HyperKZG operations.

/// Errors produced by the HyperKZG commitment scheme.
#[derive(Debug, thiserror::Error)]
pub enum HyperKZGError {
    #[error("SRS too small: have {have} powers, need {need}")]
    SrsTooSmall { have: usize, need: usize },

    #[error("proof verification failed")]
    VerificationFailed,

    #[error("invalid proof structure: {0}")]
    InvalidProof(&'static str),
}
