use jolt_witness::WitnessError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum KernelError {
    #[error(transparent)]
    Witness(#[from] WitnessError),

    #[error(transparent)]
    Openings(#[from] jolt_openings::OpeningsError),

    /// A polynomial's dimensions are incompatible with the commitment grid.
    #[error("invalid commitment geometry: {reason}")]
    InvalidGeometry { reason: String },

    /// A stream produced a chunk the kernel cannot place in the grid.
    #[error("unsupported polynomial chunk: {reason}")]
    UnsupportedChunk { reason: String },
}
