use jolt_crypto::VectorOpeningError;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum HyraxError {
    #[error("invalid Hyrax dimensions: num_vars ({num_vars}) != row_vars ({row_vars}) + col_vars ({col_vars})")]
    InvalidDimensions {
        num_vars: usize,
        row_vars: usize,
        col_vars: usize,
    },

    #[error("dimension {dimension} is too large to fit in usize")]
    DimensionTooLarge { dimension: usize },

    #[error("point length mismatch: expected {expected}, got {got}")]
    PointLengthMismatch { expected: usize, got: usize },

    #[error("polynomial variable mismatch: expected {expected}, got {got}")]
    PolynomialVariableMismatch { expected: usize, got: usize },

    #[error("row length {row_len} exceeds vector commitment capacity {capacity}")]
    CommitmentCapacityExceeded { capacity: usize, row_len: usize },

    #[error("row commitment count mismatch: expected {expected}, got {got}")]
    RowCommitmentCountMismatch { expected: usize, got: usize },

    #[error("row iteration skipped row {expected}; got row {got}")]
    RowIterationOutOfOrder { expected: usize, got: usize },

    #[error("row iteration count mismatch: expected {expected}, got {got}")]
    RowIterationCountMismatch { expected: usize, got: usize },

    #[error("row length mismatch: expected {expected}, got {got}")]
    RowLengthMismatch { expected: usize, got: usize },

    #[error("vector opening failed: {0}")]
    VectorOpening(#[from] VectorOpeningError),

    #[error("evaluation mismatch")]
    EvaluationMismatch,
}
