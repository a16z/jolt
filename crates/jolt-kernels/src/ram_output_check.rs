//! The stage-2 RAM output-check slot.

use jolt_claims::protocols::jolt::ReadWriteDimensions;
use jolt_field::Field;
use jolt_program::preprocess::PublicIoMemory;
use jolt_verifier::stages::stage2::ram_output_check::RamOutputCheck;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::{KernelError, ProofSession, SumcheckKernel};

/// The stage-2 RAM output-check slot.
pub trait RamOutputCheckProver<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        dimensions: ReadWriteDimensions,
        ram_log_k: usize,
        output_address_challenges: &[F],
        public_memory: PublicIoMemory,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamOutputCheck<F>>>, KernelError<F>>;
}
