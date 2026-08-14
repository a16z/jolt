use jolt_field::AkitaField;
use jolt_verifier::stages::stage4::registers_read_write_checking::RegistersReadWriteChecking;
use jolt_witness::JoltWitnessPlane;

use super::backend::MetalBackend;
use crate::optimized::registers_read_write::OptimizedRegistersReadWrite;
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

impl PrepareKernel<AkitaField, RegistersReadWriteChecking<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, RegistersReadWriteChecking<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = RegistersReadWriteChecking<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let prepared = OptimizedRegistersReadWrite.prepare(session, witness, inputs)?;
        super::instruction_read_raf::start_instruction_read_raf_scatter(session)?;
        Ok(prepared)
    }
}
