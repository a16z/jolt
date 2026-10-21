#[cfg(feature = "whiteout-prover")]
use jolt_crypto::sample::SampleError;
use jolt_crypto::CryptoError;
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
    /// The challenge family fails its own validation.
    #[error("invalid challenge family: {0}")]
    ChallengeFamily(&'static str),
    /// The segment table has no Gaussian-masked segment.
    #[error("the segment table has no Gaussian segment")]
    NoGaussianSegment,
    /// A width multiplier outside the sampler's `[1, 2^58]`.
    #[error("width multiplier {width} outside [1, 2^58]")]
    WidthOutOfRange { width: u64 },
    /// The width admits `‖cs‖_σ > 1/α` for an honest `s` of the table's
    /// shapes, breaking the joint rejection theorem's hypothesis.
    #[error("Gaussian width too narrow for the segments' honest norms")]
    WidthTooNarrow,
    /// The parameters exceed the joint rejection rule's exact arithmetic.
    #[error("parameters exceed the rejection rule's arithmetic bounds")]
    RejectionArithmetic,
    /// Parameter arithmetic overflowed its integer range.
    #[error("Σ-protocol parameter arithmetic overflows")]
    Overflow,
    /// The residual relation or its placement in `s` is malformed.
    #[error("invalid Whiteout relation: {0}")]
    InvalidRelation(&'static str),
    /// The prover's witness violates residual row `row` (affine rows, then
    /// scalarity rows, then product rows, then `ct(g) = 0`), checked before
    /// any challenge. The proof must abort.
    #[error("the residual witness violates row {row}")]
    ResidualUnsatisfied { row: usize },
    /// `N·s` differs from the commitment images. The proof must abort.
    #[error("the private vector does not open the commitment images")]
    ResidualOpeningMismatch,
    /// Every one of the `T_att` Σ-protocol attempts was rejected. The event
    /// is independent of the witness; the caller may rerun the whole proof
    /// with fresh randomness.
    #[error("every Σ-protocol attempt was rejected")]
    AttemptsExhausted,
    /// A release invariant failed: only a malformed witness or a bug
    /// produces it. Never retried, never recorded in the transcript.
    #[error("Whiteout invariant violated: {0}")]
    InvariantViolation(&'static str),
    /// The response statement does not hold.
    #[error("response rejected: {0}")]
    ResponseRejected(&'static str),
    /// A lattice primitive rejected its arguments.
    #[error(transparent)]
    Crypto(#[from] CryptoError),
    /// The sampling layer rejected its parameters.
    #[cfg(feature = "whiteout-prover")]
    #[error(transparent)]
    Sample(#[from] SampleError),
    /// The DIT worker pool for the attempts could not be started.
    #[cfg(feature = "whiteout-prover")]
    #[error("could not start the Σ-protocol's DIT thread pool")]
    DitPool,
}
