//! Error types for Jolt proof verification.

use jolt_spartan::SpartanError;
use jolt_sumcheck::SumcheckError;

/// Errors that can occur during Jolt proof verification or proving.
#[derive(Debug, thiserror::Error)]
pub enum JoltError {
    #[error("spartan error: {0}")]
    Spartan(#[from] SpartanError),

    #[error("sumcheck error: {0}")]
    Sumcheck(#[from] SumcheckError),

    #[error("opening verification failed: {0}")]
    Opening(String),

    #[error("stage {stage} verification failed: {reason}")]
    StageVerification { stage: usize, reason: String },

    #[error("proof deserialization failed: {0}")]
    Deserialization(String),

    #[error("invalid proof structure: {0}")]
    InvalidProof(String),
}
