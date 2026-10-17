//! CUDA backend selection using optimized compute kernels and the host witness plane.
//!
//! [`JoltBackend::cuda`] uses the optimized registry. [`CudaDoryScheme`] selects
//! the PCS independently and shares its proof and setup types with `DoryScheme`.
//! Both entry points execute entirely on the CPU.

use jolt_field::JoltField;
use jolt_openings::CommitmentScheme;

use crate::{JoltBackend, ModeStreamingCommitment};

mod common;
mod dory;

pub use common::error::CudaError;
pub use dory::CudaDoryScheme;

impl<F, PCS> JoltBackend<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    /// Select the optimized kernel slots through the CUDA backend entry point.
    pub fn cuda() -> Self
    where
        PCS: ModeStreamingCommitment,
    {
        Self::optimized()
    }
}
