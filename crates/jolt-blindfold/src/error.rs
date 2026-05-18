use jolt_crypto::VectorOpeningError;
use jolt_field::FieldCore;
use jolt_r1cs::{ClaimLoweringError, ConstraintMatrixEvalError};
use jolt_sumcheck::{SumcheckError, SumcheckR1csError};
use thiserror::Error as ThisError;

#[derive(Clone, Debug, ThisError, PartialEq, Eq)]
pub enum Error {
    #[error(transparent)]
    Layout(#[from] LayoutError),
    #[error(transparent)]
    Claim(#[from] ClaimLoweringError),
    #[error("stage {stage_index}: {source}")]
    Sumcheck {
        stage_index: usize,
        source: SumcheckR1csError,
    },
    #[error("layout has {layout_stages} stages but claims have {claim_stages}")]
    LayoutStageCountMismatch {
        claim_stages: usize,
        layout_stages: usize,
    },
}

#[derive(Clone, Debug, ThisError, PartialEq, Eq)]
pub enum LayoutError {
    #[error("claims have {claim_stages} stages but committed inputs have {input_stages}")]
    StageCountMismatch {
        claim_stages: usize,
        input_stages: usize,
    },
    #[error("stage {stage_index}: {source}")]
    Sumcheck {
        stage_index: usize,
        source: SumcheckR1csError,
    },
}

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
pub enum VerificationError<F: FieldCore> {
    #[error("claims have {claim_stages} stages but proof has {proof_stages}")]
    StageCountMismatch {
        claim_stages: usize,
        proof_stages: usize,
    },
    #[error("stage {stage_index}: {source}")]
    Sumcheck {
        stage_index: usize,
        source: SumcheckError<F>,
    },
    #[error("outer folded R1CS sumcheck: {source}")]
    OuterSumcheck { source: SumcheckError<F> },
    #[error("inner folded R1CS sumcheck: {source}")]
    InnerSumcheck { source: SumcheckError<F> },
    #[error(transparent)]
    R1cs(#[from] Error),
    #[error(transparent)]
    R1csMatrix(#[from] ConstraintMatrixEvalError),
    #[error(transparent)]
    Relaxed(#[from] RelaxedError),
    #[error(transparent)]
    VectorOpening(#[from] VectorOpeningError),
    #[error("{name} must be a non-zero power of two, got {value}")]
    InvalidPowerOfTwo { name: &'static str, value: usize },
    #[error("folded eval commitment {index} does not match opened value and blinding")]
    EvalCommitmentMismatch { index: usize },
    #[error("outer final claim mismatch: expected {expected}, got {actual}")]
    OuterFinalClaimMismatch { expected: F, actual: F },
    #[error("inner final claim mismatch: expected {expected}, got {actual}")]
    InnerFinalClaimMismatch { expected: F, actual: F },
}
