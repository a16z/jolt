use cudarc::driver::DriverError;
use jolt_field::JoltField;
use jolt_sumcheck::SumcheckError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("CUDA driver error: {0}")]
    Driver(#[from] DriverError),

    #[error("device buffer holds {got} field elements, expected {expected}")]
    LengthMismatch { expected: usize, got: usize },

    #[error("CUDA kernel not implemented: {kernel}")]
    NotImplemented { kernel: &'static str },

    #[error("CUDA kernel invariant violated: {reason}")]
    InvariantViolation { reason: &'static str },
}

pub(crate) fn backend<F: JoltField>(kind: &'static str) -> impl Fn(CudaError) -> SumcheckError<F> {
    move |error| {
        tracing::error!(operation = kind, error = %error, "CUDA round evaluation failed");
        SumcheckError::MissingEvaluationSource { kind }
    }
}
