//! Cycle phase of the two-phase committed-bytecode claim-reduction relation.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use super::BytecodeReductionShape;
use crate::protocols::jolt::geometry::claim_reductions::bytecode::{
    assert_valid_chunk_count, bytecode_val_stage_opening, cycle_phase_intermediate_opening,
    final_output_expr, NUM_BYTECODE_VAL_STAGES,
};
use crate::protocols::jolt::{
    BytecodeClaimReductionChallenge, JoltChallengeId, JoltExpr, JoltOpeningId, JoltDerivedId,
    JoltRelationId, JoltSumcheckSpec,
};
use crate::{challenge, opening, InputClaims, OutputClaims, SymbolicSumcheck};

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
#[derive(Clone, Debug, InputClaims)]
pub struct BytecodeReductionCyclePhaseInputClaims<C> {
    #[opening(BytecodeValStage, from = BytecodeReadRaf)]
    pub val_stages: Vec<C>,
}

/// Cycle phase of the committed-bytecode reduction: batches the five staged
/// `BytecodeValStage(i)` openings by powers of `eta` and reduces them to either
/// the cycle-phase intermediate opening (when an address phase follows) or the
/// committed `BytecodeChunk(i)` openings weighted by `ChunkOutputWeight`.
pub struct CyclePhase {
    shape: BytecodeReductionShape,
}

impl SymbolicSumcheck for CyclePhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = BytecodeReductionShape;

    fn new(shape: BytecodeReductionShape) -> Self {
        assert_valid_chunk_count(shape.1);
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::BytecodeClaimReductionCyclePhase
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.0.cycle_sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let eta = challenge(BytecodeClaimReductionChallenge::Eta);
        let mut input = JoltExpr::zero();
        for stage in 0..NUM_BYTECODE_VAL_STAGES {
            input = input + eta.clone().pow(stage) * opening(bytecode_val_stage_opening(stage));
        }
        input
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let (dimensions, chunk_count) = self.shape;
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
    use crate::protocols::jolt::geometry::claim_reductions::bytecode::final_bytecode_chunk_opening;
    use crate::protocols::jolt::{BytecodeClaimReductionPublic, PrecommittedReductionDimensions};
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

        let cycle = CyclePhase::new((dimensions, 2));
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
        let relation = CyclePhase::new((dimensions, 2));

        assert_eq!(
            CyclePhase::id(),
            JoltRelationId::BytecodeClaimReductionCyclePhase
        );
        assert_eq!(relation.spec(), dimensions.cycle_sumcheck());
        assert_eq!(
            relation.input_expression::<Fr>().required_openings(),
            (0..NUM_BYTECODE_VAL_STAGES)
                .map(bytecode_val_stage_opening)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            relation.output_expression::<Fr>().required_openings(),
            vec![cycle_phase_intermediate_opening()]
        );
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![JoltChallengeId::from(BytecodeClaimReductionChallenge::Eta)]
        );
        assert!(relation.required_deriveds::<Fr>().is_empty());
    }

    #[test]
    fn cycle_phase_without_address_phase_opens_committed_chunks() {
        let dimensions = PrecommittedReductionDimensions::new(4, 3, false);
        let relation = CyclePhase::new((dimensions, 2));

        assert_eq!(
            relation.output_expression::<Fr>().required_openings(),
            vec![
                final_bytecode_chunk_opening(0),
                final_bytecode_chunk_opening(1),
            ]
        );
        assert_eq!(
            relation.required_deriveds::<Fr>(),
            vec![
                JoltDerivedId::from(BytecodeClaimReductionPublic::ChunkOutputWeight(0)),
                JoltDerivedId::from(BytecodeClaimReductionPublic::ChunkOutputWeight(1)),
            ]
        );
    }
}
