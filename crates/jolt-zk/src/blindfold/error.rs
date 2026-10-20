use jolt_crypto::VectorOpeningError;
use jolt_field::JoltField;
use jolt_r1cs::ConstraintMatrixEvalError;
use jolt_sumcheck::SumcheckError;
use jolt_transcript::TranscriptError;
use thiserror::Error as ThisError;

use crate::masking::ZkError;

#[derive(Clone, Debug, ThisError, PartialEq, Eq)]
pub enum RelaxedError {
    #[error("{name} length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{name} dimension {value} cannot be represented as a power-of-two size")]
    DimensionOverflow { name: &'static str, value: usize },
    #[error("{name} dimensions are inconsistent: total {total} is smaller than used {used}")]
    InconsistentDimensions {
        name: &'static str,
        total: usize,
        used: usize,
    },
}

#[derive(Debug, ThisError)]
pub enum ProverError<F: JoltField> {
    #[error(transparent)]
    Relaxed(#[from] RelaxedError),
    #[error(transparent)]
    R1csMatrix(#[from] ConstraintMatrixEvalError),
    #[error(transparent)]
    VectorOpening(#[from] VectorOpeningError),
    #[error("{name} length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("witness row {row} length mismatch: expected {expected}, got {actual}")]
    WitnessRowLengthMismatch {
        row: usize,
        expected: usize,
        actual: usize,
    },
    #[error("{name} row length {row_len} exceeds vector commitment capacity {capacity}")]
    CommitmentCapacityExceeded {
        name: &'static str,
        capacity: usize,
        row_len: usize,
    },
    #[error("{name} row commitment backend failed: {reason}")]
    RowCommitmentBackend { name: &'static str, reason: String },
    #[error("{name} backend kernel failed: {reason}")]
    BackendKernel { name: &'static str, reason: String },
    #[error("{name} must be a non-zero power of two, got {value}")]
    InvalidPowerOfTwo { name: &'static str, value: usize },
    #[error("{name} dimension {value} cannot be represented")]
    DimensionOverflow { name: &'static str, value: usize },
    #[error("folded eval commitment {index} does not match opened value and blinding")]
    EvalCommitmentMismatch { index: usize },
    #[error("folded eval witness {kind} {index} does not match opened witness coordinate: expected {expected}, got {actual}")]
    EvalWitnessMismatch {
        kind: &'static str,
        index: usize,
        expected: F,
        actual: F,
    },
    #[error("interpolation denominator is zero for degree {degree} at point {point}")]
    ZeroInterpolationDenominator { degree: usize, point: usize },
    #[error("sumcheck round claim mismatch: expected {expected}, got {actual}")]
    SumcheckRoundClaimMismatch { expected: F, actual: F },
    #[error("multilinear evaluation length mismatch: expected {expected}, got {actual}")]
    MultilinearLengthMismatch { expected: usize, actual: usize },
    #[error("{name} must have at least one sumcheck round")]
    DegenerateSumcheck { name: &'static str },
    #[error("BlindFold sumcheck: {0}")]
    Sumcheck(#[from] SumcheckError<F>),
}

#[derive(Debug, ThisError)]
pub enum VerificationError<F: JoltField> {
    #[error("BlindFold proof transcript: {0}")]
    Transcript(#[from] TranscriptError),
    #[error("outer folded R1CS sumcheck: {source}")]
    OuterSumcheck { source: SumcheckError<F> },
    #[error("inner folded R1CS sumcheck: {source}")]
    InnerSumcheck { source: SumcheckError<F> },
    #[error(transparent)]
    R1csMatrix(#[from] ConstraintMatrixEvalError),
    #[error(transparent)]
    Relaxed(#[from] RelaxedError),
    #[error(transparent)]
    VectorOpening(#[from] VectorOpeningError),
    #[error("{name} must be a non-zero power of two, got {value}")]
    InvalidPowerOfTwo { name: &'static str, value: usize },
    #[error("{name} must have at least one sumcheck round")]
    DegenerateSumcheck { name: &'static str },
    #[error("folded eval commitment {index} does not match opened value and blinding")]
    EvalCommitmentMismatch { index: usize },
    #[error("folded eval witness {kind} {index} does not match opened witness coordinate")]
    EvalWitnessMismatch { kind: &'static str, index: usize },
    #[error("folded eval witness {kind} {index} opening has a non-zero value outside the dedicated slot")]
    EvalWitnessRowNotDedicated { kind: &'static str, index: usize },
    #[error("outer final claim mismatch: expected {expected}, got {actual}")]
    OuterFinalClaimMismatch { expected: F, actual: F },
    #[error("inner final claim mismatch: expected {expected}, got {actual}")]
    InnerFinalClaimMismatch { expected: F, actual: F },
}

impl<F: JoltField + 'static> From<ProverError<F>> for ZkError {
    fn from(error: ProverError<F>) -> Self {
        Self::BlindFoldProver(Box::new(error))
    }
}

/// Truncation inside the BlindFold proof surfaces as
/// [`ZkError::Transcript`], as it does everywhere else in a proof.
impl<F: JoltField + 'static> From<VerificationError<F>> for ZkError {
    fn from(error: VerificationError<F>) -> Self {
        if let VerificationError::Transcript(error)
        | VerificationError::OuterSumcheck {
            source: SumcheckError::Transcript(error),
        }
        | VerificationError::InnerSumcheck {
            source: SumcheckError::Transcript(error),
        } = error
        {
            return Self::Transcript(error);
        }
        Self::BlindFoldRejected(Box::new(error))
    }
}
