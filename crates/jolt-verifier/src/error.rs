//! Error types for Jolt proof verification.

use jolt_openings::OpeningsError;
use jolt_spartan::SpartanError;
use jolt_sumcheck::SumcheckError;

/// Errors that can occur during Jolt proof verification.
#[derive(Debug, thiserror::Error)]
pub enum JoltError {
    #[error("spartan error: {0}")]
    Spartan(#[from] SpartanError),

    #[error("sumcheck error: {0}")]
    Sumcheck(#[from] SumcheckError),

    #[error("opening verification failed: {0}")]
    Opening(#[from] OpeningsError),

    #[error("stage {stage} verification failed: {reason}")]
    StageVerification { stage: usize, reason: String },

    #[error("invalid proof structure: {0}")]
    InvalidProof(String),

    #[error("evaluation check failed at stage {stage}: {reason}")]
    EvaluationMismatch { stage: usize, reason: String },
}
