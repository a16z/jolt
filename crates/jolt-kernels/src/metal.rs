use jolt_akita::AkitaField;
use jolt_metal::MetalError;
use jolt_openings::CommitmentScheme;

use crate::optimized::spartan_outer::metal::MetalOuterRemainder;
use crate::JoltBackend;

mod trace_commitment;
pub use trace_commitment::MetalTraceCommitment;

impl<PCS: CommitmentScheme<Field = AkitaField>> JoltBackend<AkitaField, PCS> {
    /// Use Metal for large Spartan outer remainder rounds, retaining the
    /// current slots for preparation, commitments, openings and other rounds.
    /// Compiles the shaders before any proof starts; device/setup errors are
    /// returned to the caller rather than silently selecting the CPU backend.
    pub fn with_metal_compute(mut self) -> Result<Self, MetalError> {
        self.spartan_outer_remainder = Box::new(MetalOuterRemainder::new()?);
        Ok(self)
    }
}
