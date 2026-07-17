//! The instruction input-virtualization (stage 3) slot.

use jolt_claims::protocols::jolt::relations::instruction::InstructionInputChallenges;
use jolt_claims::protocols::jolt::TraceDimensions;
use jolt_field::Field;
use jolt_verifier::stages::stage3::outputs::InstructionInput;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::{KernelError, ProofSession, SumcheckKernel};

/// The stage-3 instruction input-virtualization slot.
pub trait InstructionInputProver<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        trace_dimensions: TraceDimensions,
        product_remainder_point: &[F],
        challenges: &InstructionInputChallenges<F>,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionInput<F>>>, KernelError<F>>;
}
