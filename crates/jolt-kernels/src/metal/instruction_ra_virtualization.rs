use jolt_field::AkitaField;
use jolt_verifier::stages::stage6b::instruction_ra_virtualization::InstructionRaVirtualization;
use jolt_witness::JoltWitnessPlane;

use super::instruction_read_raf::MetalBackend;
use super::solinas::ResidentLookupIndexPlane;
use crate::optimized::instruction_ra_virtualization::prepare_optimized_instruction_ra_virtualization;
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstructionRaVirtualizationMetalConfig {
    pub trace_cutoff_elements: usize,
}

impl Default for InstructionRaVirtualizationMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 18,
        }
    }
}

impl PrepareKernel<AkitaField, InstructionRaVirtualization<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, InstructionRaVirtualization<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = InstructionRaVirtualization<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let trace_elements = 1usize << inputs.relation.dimensions().log_t();
        let cpu = prepare_optimized_instruction_ra_virtualization(session, witness, inputs)?;
        if trace_elements
            < self
                .config
                .instruction_ra_virtualization
                .trace_cutoff_elements
        {
            return Ok(Box::new(cpu));
        }

        let plane =
            session
                .take::<ResidentLookupIndexPlane>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "Metal Instruction RA requires the resident stage-5 lookup plane",
                })?;
        if plane.len() != trace_elements {
            return Err(KernelError::TableSizeMismatch {
                table: "resident Metal lookup-index plane".to_owned(),
                expected: trace_elements,
                got: plane.len(),
            });
        }
        if plane.device_registry_id() != self.context.device_registry_id() {
            return Err(KernelError::InvariantViolation {
                reason: "resident Metal lookup-index plane belongs to another device",
            });
        }

        drop(plane);
        Ok(Box::new(cpu))
    }
}
