#[cfg(feature = "whiteout-prover")]
use jolt_crypto::sample::SampleError;
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
    /// A segment of the private vector has an unusable shape.
    #[error("segment {index} is malformed: {reason}")]
    Segment { index: usize, reason: &'static str },
    /// The private vector's length differs from its segment table's.
    #[error("the private vector has {actual} coefficients, its segments {expected}")]
    WitnessLength { expected: usize, actual: usize },
    /// A coefficient of the private vector lies outside its segment's
    /// alphabet or shape. Only a malformed witness produces it; the proof
    /// must abort.
    #[error("a coefficient of the private vector is outside its segment's alphabet or shape")]
    WitnessOutsideAlphabet,
    /// The Σ-protocol parameters fail a check of Whiteout §6.2.
    #[error("Σ-protocol parameters rejected: {0}")]
    Parameters(&'static str),
    /// The sampling layer rejected its parameters.
    #[cfg(feature = "whiteout-prover")]
    #[error(transparent)]
    Sample(#[from] SampleError),
}
