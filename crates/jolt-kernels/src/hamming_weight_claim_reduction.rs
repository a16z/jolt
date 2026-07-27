//! The Hamming-weight claim-reduction (stage 7) slot.

use jolt_claims::protocols::jolt::geometry::claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions;
use jolt_claims::protocols::jolt::relations::claim_reductions::hamming_weight::HammingWeightClaimReductionChallenges;
use jolt_field::Field;
use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::HammingWeightClaimReduction;
use jolt_witness::JoltWitnessOracle;

use crate::{KernelError, ProofSession, ProveSumcheck};

/// The stage-7 Hamming-weight claim-reduction slot. `r_cycle` and `r_address`
/// are the stage-6b booleanity opening point's splits; `virtualization_points`
/// are the leading chunk coordinates of each stage-6b RA virtualization
/// opening point, in canonical layout order.
pub trait HammingWeightClaimReductionProver<F: Field> {
    #[expect(
        clippy::too_many_arguments,
        reason = "the relation's construction data"
    )]
    fn prepare(
        &self,
        session: &mut ProofSession,
        dimensions: HammingWeightClaimReductionDimensions,
        r_cycle: &[F],
        r_address: &[F],
        virtualization_points: &[Vec<F>],
        challenges: &HammingWeightClaimReductionChallenges<F>,
        witness: &dyn JoltWitnessOracle<F>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = HammingWeightClaimReduction<F>>>, KernelError<F>>;
}
