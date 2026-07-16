//! The registers claim-reduction (stage 3) slot.

use jolt_claims::protocols::jolt::relations::claim_reductions::registers::RegistersClaimReductionChallenges;
use jolt_claims::protocols::jolt::TraceDimensions;
use jolt_field::Field;
use jolt_verifier::stages::stage3::outputs::RegistersClaimReduction;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::{KernelError, ProofSession, ProveSumcheck};

/// The stage-3 registers claim-reduction slot.
pub trait RegistersClaimReductionProver<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        trace_dimensions: TraceDimensions,
        product_uniskip_tau_low: &[F],
        challenges: &RegistersClaimReductionChallenges<F>,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = RegistersClaimReduction<F>>>, KernelError<F>>;
}
