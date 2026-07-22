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
    #[error("stage {stage}: missing {component}")]
    MissingStageComponent {
        stage: String,
        component: &'static str,
    },
    #[error("{name} must be non-zero when committed rows are present")]
    MissingRowLength { name: &'static str },
    #[error("{name} has {ids} opening ids but only {slots} committed row slots")]
    OpeningRowCapacityExceeded {
        name: &'static str,
        ids: usize,
        slots: usize,
    },
    #[error("final opening binding must reference at least one opening")]
    EmptyFinalOpeningBinding,
    #[error("{name} row count mismatch: expected {expected}, got {actual}")]
    CommittedRowCountMismatch {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("opening id appears more than once")]
    DuplicateOpeningSource,
    #[error("opening alias refers to an unknown source")]
    MissingOpeningAliasSource,
    #[error("stage {stage_index}: {source}")]
    Sumcheck {
        stage_index: usize,
        source: SumcheckR1csError,
    },
    #[error("layout has {layout_stages} stages but statement has {statement_stages}")]
    LayoutStageCountMismatch {
        statement_stages: usize,
        layout_stages: usize,
    },
}

#[derive(Clone, Debug, ThisError, PartialEq, Eq)]
pub enum LayoutError {
    #[error("stage {stage_index}: {source}")]
    Sumcheck {
        stage_index: usize,
        source: SumcheckR1csError,
    },
    #[error("{name} dimension {value} cannot be represented as a power-of-two size")]
    DimensionOverflow { name: &'static str, value: usize },
    #[error("{name} must be non-zero when committed rows are present")]
    MissingRowLength { name: &'static str },
    #[error("{name} has {ids} opening ids but only {slots} committed row slots")]
    OpeningRowCapacityExceeded {
        name: &'static str,
        ids: usize,
        slots: usize,
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
pub enum ProverError<F: FieldCore> {
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
    #[error(transparent)]
    Statement(#[from] Error),
    #[error("witness assignment: {0}")]
    Assignment(#[from] jolt_r1cs::R1csBuilderError),
    #[error("witness assignment: {0}")]
    Domain(#[from] SumcheckError<F>),
    #[error("stage {stage_index} witness {name} mismatch: expected {expected}, got {actual}")]
    StageWitnessShape {
        stage_index: usize,
        name: &'static str,
        expected: usize,
        actual: usize,
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
