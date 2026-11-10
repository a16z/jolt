use jolt_akita::AkitaField;
use jolt_metal::MetalError;
use jolt_openings::CommitmentScheme;

use crate::instruction_rows::InstructionCycleRow;
use crate::optimized::booleanity::metal::MetalBooleanity;
use crate::optimized::bytecode_read_raf::metal::MetalBytecodeReadRafAddress;
use crate::optimized::instruction_input::metal::MetalInstructionInput;
use crate::optimized::instruction_ra_virtualization::metal::MetalInstructionRa;
use crate::optimized::instruction_read_raf::metal::MetalInstructionReadRaf;
use crate::{
    CommitWitness, JoltBackend, KernelError, ProofSession, WitnessCommitRequest, WitnessCommitment,
};
use jolt_witness::JoltWitnessPlane;

mod trace_compute;
pub use trace_compute::MetalTraceCompute;

impl<PCS: CommitmentScheme<Field = AkitaField>> JoltBackend<AkitaField, PCS> {
    /// Use Metal for admitted instruction, bytecode and Booleanity kernels.
    /// The commitment slot also shares a bounded canonical CPU row projection
    /// with later instruction kernels; other slots retain their implementations.
    /// Compiles the shaders before any proof starts; device/setup errors are
    /// returned to the caller rather than silently selecting the CPU backend.
    pub fn with_metal_compute(mut self) -> Result<Self, MetalError> {
        self.commit = Box::new(SharedInstructionCommit {
            inner: self.commit,
            byte_limit: 6 << 30,
        });
        self.instruction_input = Box::new(MetalInstructionInput::new()?);
        self.instruction_read_raf = Box::new(MetalInstructionReadRaf::new()?);
        self.instruction_ra_virtualization = Box::new(MetalInstructionRa::new()?);
        self.booleanity_cycle = Box::new(MetalBooleanity::new()?);
        let bytecode = MetalBytecodeReadRafAddress::new()?;
        self.bytecode_read_raf_cycle = Box::new(bytecode.cycle());
        self.bytecode_read_raf_address = Box::new(bytecode);
        Ok(self)
    }
}

/// Keep the early CPU projection bounded independently of the GPU selector
/// cache. Six GiB admits the five-GiB T27 rows and declines larger projections.
struct SharedInstructionCommit<PCS: CommitmentScheme<Field = AkitaField>> {
    inner: Box<dyn CommitWitness<AkitaField, PCS>>,
    byte_limit: u64,
}

impl<PCS: CommitmentScheme<Field = AkitaField>> CommitWitness<AkitaField, PCS>
    for SharedInstructionCommit<PCS>
{
    fn commit_witness(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        request: WitnessCommitRequest<'_, PCS>,
    ) -> Result<WitnessCommitment<PCS>, KernelError<AkitaField>> {
        InstructionCycleRow::prepare_for_commit(
            session,
            witness,
            1 << request.shape().log_t,
            self.byte_limit,
        )?;
        self.inner.commit_witness(session, witness, request)
    }
}
