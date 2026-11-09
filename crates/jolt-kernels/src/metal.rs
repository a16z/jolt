use jolt_akita::AkitaField;
use jolt_metal::MetalError;
use jolt_openings::CommitmentScheme;

use crate::optimized::booleanity::metal::MetalBooleanity;
use crate::optimized::bytecode_read_raf::metal::MetalBytecodeReadRafAddress;
use crate::optimized::instruction_ra_virtualization::metal::MetalInstructionRa;
use crate::optimized::instruction_read_raf::metal::MetalInstructionReadRaf;
use crate::JoltBackend;

mod trace_compute;
pub use trace_compute::MetalTraceCompute;

impl<PCS: CommitmentScheme<Field = AkitaField>> JoltBackend<AkitaField, PCS> {
    /// Use Metal for compact ReadRAF, instruction-RA and Booleanity cycle rounds.
    /// Other slots retain their current implementations.
    /// Compiles the shaders before any proof starts; device/setup errors are
    /// returned to the caller rather than silently selecting the CPU backend.
    pub fn with_metal_compute(mut self) -> Result<Self, MetalError> {
        self.instruction_read_raf = Box::new(MetalInstructionReadRaf::new()?);
        self.instruction_ra_virtualization = Box::new(MetalInstructionRa::new()?);
        self.booleanity_cycle = Box::new(MetalBooleanity::new()?);
        self.bytecode_read_raf_address = Box::new(MetalBytecodeReadRafAddress::new()?);
        Ok(self)
    }
}
