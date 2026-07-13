//! The instruction RA virtualization slot: stage 6b.

use jolt_claims::protocols::jolt::geometry::instruction::InstructionRaVirtualizationDimensions;
use jolt_claims::protocols::jolt::relations::instruction::InstructionRaVirtualizationChallenges;
use jolt_field::Field;
use jolt_verifier::stages::stage6b::instruction_ra_virtualization::InstructionRaVirtualization;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::{KernelError, ProofSession, ProveSumcheck};

/// The stage-6b instruction RA virtualization slot.
pub trait InstructionRaVirtualizationProver<F: Field> {
    #[expect(
        clippy::too_many_arguments,
        reason = "the relation's construction data"
    )]
    fn prepare(
        &self,
        session: &mut ProofSession,
        dimensions: InstructionRaVirtualizationDimensions,
        instruction_r_address: &[F],
        instruction_r_cycle: &[F],
        committed_chunk_bits: usize,
        challenges: &InstructionRaVirtualizationChallenges<F>,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = InstructionRaVirtualization<F>>>, KernelError<F>>;
}
