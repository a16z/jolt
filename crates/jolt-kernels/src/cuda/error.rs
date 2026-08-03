use cudarc::driver::DriverError;
use cudarc::nvrtc::CompileError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("CUDA driver error: {0}")]
    Driver(#[from] DriverError),

    #[error("CUDA kernel compilation failed: {0}")]
    Compile(#[from] Box<CompileError>),

    #[error("no usable CUDA device: {reason}")]
    NoDevice { reason: String },

    #[error("device buffer holds {got} field elements, expected {expected}")]
    LengthMismatch { expected: usize, got: usize },

    #[error("device buffer of {len} field elements is not {limbs}-limb aligned")]
    LimbAlignment { len: usize, limbs: usize },

    #[error("CUDA kernel not implemented: {kernel}")]
    NotImplemented { kernel: &'static str },
}

impl From<CompileError> for CudaError {
    fn from(error: CompileError) -> Self {
        Self::Compile(Box::new(error))
    }
}
