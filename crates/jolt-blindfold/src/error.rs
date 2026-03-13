//! Error types for the BlindFold protocol.

/// Errors that can occur during BlindFold proof generation or verification.
#[derive(Debug, thiserror::Error)]
pub enum BlindFoldError {
    /// The number of stages provided does not match what the accumulator holds.
    #[error("stage count mismatch: expected {expected}, got {actual}")]
    StageCountMismatch { expected: usize, actual: usize },

    /// The relaxed Spartan proof over the folded instance failed.
    #[error("relaxed Spartan failed: {0}")]
    Spartan(#[from] jolt_spartan::SpartanError),
}
