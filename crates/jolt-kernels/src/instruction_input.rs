//! The instruction input-virtualization (stage 3) slot.

use jolt_claims::protocols::jolt::relations::instruction::InstructionInputChallenges;
use jolt_claims::protocols::jolt::TraceDimensions;
use jolt_field::Field;
use jolt_verifier::stages::stage3::outputs::InstructionInput;
use jolt_witness::JoltWitnessOracle;

use crate::{KernelError, ProofSession, ProveSumcheck};

/// The stage-3 instruction input-virtualization slot.
pub trait InstructionInputProver<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        trace_dimensions: TraceDimensions,
        product_remainder_point: &[F],
        challenges: &InstructionInputChallenges<F>,
        witness: &dyn JoltWitnessOracle<F>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = InstructionInput<F>>>, KernelError<F>>;
}
