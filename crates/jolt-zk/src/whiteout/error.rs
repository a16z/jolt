use thiserror::Error as ThisError;

/// Malformed inputs to the Whiteout building blocks.
#[derive(Clone, Debug, ThisError, PartialEq, Eq)]
pub enum WhiteoutError {
    /// The coefficient count is not a multiple of the ring degree `D`, or `D`
    /// is zero.
    #[error("a lifted functional needs a multiple of D = {degree} coefficients, got {actual}")]
    FunctionalLength { degree: usize, actual: usize },
    /// The ring vector's length differs from the functional's.
    #[error("ring vector length mismatch: expected {expected}, got {actual}")]
    RingVectorLength { expected: usize, actual: usize },
}
