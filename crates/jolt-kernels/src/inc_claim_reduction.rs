//! The increment claim-reduction (stage 6b) slot.

use jolt_claims::protocols::jolt::relations::claim_reductions::increments::IncClaimReductionChallenges;
use jolt_claims::protocols::jolt::TraceDimensions;
use jolt_field::Field;
use jolt_verifier::stages::stage6b::inc_claim_reduction::IncClaimReduction;
use jolt_witness::JoltWitnessOracle;

use crate::{KernelError, ProofSession, ProveSumcheck};

/// The stage-6b increment claim-reduction slot. The four cycle points are the
/// upstream sources in relation order: RAM read-write, RAM val-check,
/// registers read-write, registers val-evaluation.
pub trait IncClaimReductionProver<F: Field> {
    fn prepare(
        &self,
        session: &mut ProofSession,
        trace_dimensions: TraceDimensions,
        cycle_points: &[Vec<F>; 4],
        challenges: &IncClaimReductionChallenges<F>,
        witness: &dyn JoltWitnessOracle<F>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = IncClaimReduction<F>>>, KernelError<F>>;
}
