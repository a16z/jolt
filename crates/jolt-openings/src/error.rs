use jolt_transcript::VerificationError;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum OpeningsError {
    #[error("opening proof verification failed")]
    VerificationFailed,

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

    /// The transcript refused a challenge draw: the codec does not support the
    /// request, or its rejection pool ran out.
    #[error("transcript rejected a challenge draw")]
    TranscriptRejected,
}

impl From<VerificationError> for OpeningsError {
    fn from(_: VerificationError) -> Self {
        Self::TranscriptRejected
    }
}
