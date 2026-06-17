use jolt_field::Field;
use jolt_sumcheck::SumcheckError;

#[derive(Debug, thiserror::Error)]
pub enum ProverError<F: Field> {
    #[error("batched sumcheck spec must contain at least one instance")]
    EmptyBatch,
    #[error("round {round} sumcheck invariant failed: expected {expected:?}, got {got:?}")]
    RoundCheckFailed { round: usize, expected: F, got: F },
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error(transparent)]
    Sumcheck(#[from] SumcheckError<F>),
}

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("backend does not support relation {label}")]
    UnsupportedRelation { label: &'static str },
    #[error("backend returned {got} round polynomials for {expected} active instances")]
    RoundPolynomialCountMismatch { expected: usize, got: usize },
    #[error("active instance index {index} out of range for batch of size {batch_size}")]
    InvalidActiveIndex { index: usize, batch_size: usize },
}
