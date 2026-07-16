//! The RAM RA claim-reduction slot: the stage-5 reduction of the three
//! upstream `RamRa` openings.

use jolt_claims::protocols::jolt::relations::ram::{
    RamRaClaimReductionChallenges, RamRaClaimReductionInputClaims,
};
use jolt_claims::protocols::jolt::TraceDimensions;
use jolt_field::Field;
use jolt_verifier::stages::stage5::ram_ra_claim_reduction::RamRaClaimReduction;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use crate::{KernelError, ProofSession, ProveSumcheck};

/// The stage-5 RAM RA claim-reduction slot.
pub trait RamRaClaimReductionProver<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        trace_dimensions: TraceDimensions,
        ram_log_k: usize,
        input_points: &RamRaClaimReductionInputClaims<Vec<F>>,
        challenges: &RamRaClaimReductionChallenges<F>,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = RamRaClaimReduction<F>>>, KernelError<F>>;
}
