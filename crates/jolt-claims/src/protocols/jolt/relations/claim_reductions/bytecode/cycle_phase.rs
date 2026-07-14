//! Cycle phase of the two-phase committed-bytecode claim-reduction relation.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use super::BytecodeReductionCycleShape;
use crate::protocols::jolt::geometry::claim_reductions::bytecode::{
    assert_valid_chunk_count, bytecode_val_stage_opening, cycle_phase_intermediate_opening,
    final_output_expr, NUM_BYTECODE_VAL_STAGES,
};
use crate::protocols::jolt::geometry::claim_reductions::precommitted::TWO_PHASE_DEGREE_BOUND;
use crate::protocols::jolt::{
    BytecodeClaimReductionChallenge, JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId,
    JoltRelationId,
};
use crate::{challenge, opening, InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck};

/// The produced bytecode-reduction openings: the intermediate when an address
/// phase follows, else the per-chunk final `BytecodeChunk` openings.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(BytecodeClaimReductionCyclePhase)]
pub struct BytecodeReductionCyclePhaseOutputClaims<C> {
    #[opening(BytecodeClaimReductionIntermediate)]
    pub intermediate: Option<C>,
    #[opening(committed = BytecodeChunk)]
    pub chunks: Vec<C>,
}

/// The consumed staged `BytecodeValStage` openings from the bytecode read-RAF
/// address phase.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct BytecodeReductionCyclePhaseInputClaims<C> {
    #[opening(BytecodeValStage, from = BytecodeReadRaf)]
    pub val_stages: Vec<C>,
}

/// Fiat-Shamir challenge drawn by the committed-bytecode reduction cycle phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
pub struct BytecodeReductionCyclePhaseChallenges<F> {
    #[challenge(BytecodeClaimReductionChallenge::Eta)]
    pub eta: F,
}

/// Cycle phase of the committed-bytecode reduction: batches the staged
/// `BytecodeValStage(i)` openings by powers of `eta` and reduces them to either
/// the cycle-phase intermediate opening (when an address phase follows) or the
/// committed `BytecodeChunk(i)` openings weighted by `ChunkOutputWeight`.
pub struct CyclePhase {
    shape: BytecodeReductionCycleShape,
}

impl SymbolicSumcheck for CyclePhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = BytecodeReductionCycleShape;
    type Challenges<F> = BytecodeReductionCyclePhaseChallenges<F>;
    type Inputs<C> = BytecodeReductionCyclePhaseInputClaims<C>;
    type Outputs<C> = BytecodeReductionCyclePhaseOutputClaims<C>;

    fn new(shape: BytecodeReductionCycleShape) -> Self {
        assert_valid_chunk_count(shape.1);
        assert!(
            shape.2 == NUM_BYTECODE_VAL_STAGES || shape.2 == NUM_BYTECODE_VAL_STAGES + 1,
            "bytecode reduction folds five (base) or six (lattice) staged vals"
        );
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::BytecodeClaimReductionCyclePhase
    }

    fn rounds(&self) -> usize {
        self.shape.0.cycle_phase_total_rounds()
    }

    fn degree(&self) -> usize {
        TWO_PHASE_DEGREE_BOUND
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let eta = challenge(BytecodeClaimReductionChallenge::Eta);
        let mut input = JoltExpr::zero();
        for stage in 0..self.shape.2 {
            input = input + eta.clone().pow(stage) * opening(bytecode_val_stage_opening(stage));
        }
        input
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let (dimensions, chunk_count, _) = self.shape;
        if dimensions.has_address_phase() {
            opening(cycle_phase_intermediate_opening())
        } else {
            final_output_expr(chunk_count)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::protocols::jolt::{BooleanityChallenge, PrecommittedReductionDimensions};
    use jolt_field::{Fr, FromPrimitiveInt};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    #[test]
    fn cycle_phase_batches_staged_openings_by_eta() {
        let dimensions = PrecommittedReductionDimensions::new(4, 3, true);
        let eta = fr(31);
        let stage_claims = [fr(3), fr(5), fr(7), fr(11), fr(13)];
        let zero = fr(0);

        let cycle = CyclePhase::new((dimensions, 2, NUM_BYTECODE_VAL_STAGES));
        let input = cycle.input_expression::<Fr>().evaluate(
            |id| {
                (0..NUM_BYTECODE_VAL_STAGES)
                    .find(|&stage| *id == bytecode_val_stage_opening(stage))
                    .map_or(zero, |stage| stage_claims[stage])
            },
            |id| match *id {
                JoltChallengeId::BytecodeClaimReduction(BytecodeClaimReductionChallenge::Eta) => {
                    eta
                }
                _ => zero,
            },
            |_| zero,
        );
        let mut expected_input = zero;
        let mut eta_power = fr(1);
        for claim in stage_claims {
            expected_input += eta_power * claim;
            eta_power *= eta;
        }
        assert_eq!(input, expected_input);
    }

    #[test]
    fn cycle_phase_with_address_phase_exposes_expected_dependencies() {
        let dimensions = PrecommittedReductionDimensions::new(4, 3, true);
        let relation = CyclePhase::new((dimensions, 2, NUM_BYTECODE_VAL_STAGES));

        assert_eq!(
            CyclePhase::id(),
            JoltRelationId::BytecodeClaimReductionCyclePhase
        );
        assert_eq!(relation.rounds(), dimensions.cycle_phase_total_rounds());
        assert_eq!(relation.degree(), TWO_PHASE_DEGREE_BOUND);
    }

    #[test]
    fn challenges_resolve_eta_and_miss_others() {
        let challenges = BytecodeReductionCyclePhaseChallenges { eta: fr(31) };

        assert_eq!(
            challenges
                .resolve_challenge(&JoltChallengeId::from(BytecodeClaimReductionChallenge::Eta)),
            Some(fr(31)),
        );
        assert_eq!(
            challenges.resolve_challenge(&JoltChallengeId::from(BooleanityChallenge::Gamma)),
            None,
        );
    }
}
