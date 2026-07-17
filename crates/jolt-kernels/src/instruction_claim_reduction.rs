//! The instruction claim-reduction slot: the stage-2 reduction over the
//! cycle domain.

use jolt_claims::protocols::jolt::relations::claim_reductions::instruction::InstructionClaimReductionChallenges;
use jolt_claims::protocols::jolt::TraceDimensions;
use jolt_field::Field;
use jolt_verifier::stages::stage2::instruction_claim_reduction::InstructionClaimReduction;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::{KernelError, ProofSession, SumcheckKernel};

/// The stage-2 instruction claim-reduction slot.
pub trait InstructionClaimReductionProver<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        trace_dimensions: TraceDimensions,
        tau_low: &[F],
        challenges: &InstructionClaimReductionChallenges<F>,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionClaimReduction<F>>>, KernelError<F>>;
}
