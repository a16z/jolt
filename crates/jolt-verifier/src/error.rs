use jolt_openings::OpeningsError;
use jolt_sumcheck::SumcheckError;

#[derive(Debug, thiserror::Error)]
pub enum JoltError {
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
