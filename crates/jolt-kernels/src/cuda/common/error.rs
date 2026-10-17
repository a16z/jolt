use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("CUDA kernel invariant violated: {reason}")]
    InvariantViolation { reason: &'static str },
}
