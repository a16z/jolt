use cudarc::driver::DriverError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("CUDA driver error: {0}")]
    Driver(#[from] DriverError),

    #[error("device buffer holds {got} field elements, expected {expected}")]
    LengthMismatch { expected: usize, got: usize },

    #[error("CUDA kernel invariant violated: {reason}")]
    InvariantViolation { reason: &'static str },
}
