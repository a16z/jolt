//! Address phase of the two-phase committed-bytecode claim-reduction relation.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use super::BytecodeReductionShape;
use crate::protocols::jolt::geometry::claim_reductions::bytecode::{
    assert_valid_chunk_count, cycle_phase_intermediate_opening, final_output_expr,
};
use crate::protocols::jolt::geometry::claim_reductions::precommitted::TWO_PHASE_DEGREE_BOUND;
use crate::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId,
};
use crate::{opening, InputClaims, OutputClaims, SymbolicSumcheck};

/// Produced per-chunk `BytecodeChunk(i)` openings, all sharing the reduction's
/// final opening point. Generic over the cell.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(BytecodeClaimReduction)]
pub struct BytecodeReductionAddressPhaseOutputClaims<C> {
    #[opening(committed = BytecodeChunk)]
    pub chunks: Vec<C>,
}

/// Consumed intermediate opening from the stage-6b bytecode cycle phase.
#[derive(Clone, Debug, PartialEq, Eq, InputClaims)]
pub struct BytecodeReductionAddressPhaseInputClaims<C> {
    #[opening(BytecodeClaimReductionIntermediate, from = BytecodeClaimReductionCyclePhase)]
    pub cycle_phase_intermediate: C,
}

/// Address phase of the committed-bytecode reduction: reduces the cycle-phase
/// intermediate opening to the committed `BytecodeChunk(i)` openings weighted by
/// `ChunkOutputWeight`.
pub struct AddressPhase {
    shape: BytecodeReductionShape,
}

impl SymbolicSumcheck for AddressPhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = BytecodeReductionShape;
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = BytecodeReductionAddressPhaseInputClaims<C>;
    type Outputs<C> = BytecodeReductionAddressPhaseOutputClaims<C>;

    fn new(shape: BytecodeReductionShape) -> Self {
        assert_valid_chunk_count(shape.1);
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::BytecodeClaimReduction
    }

    fn rounds(&self) -> usize {
        self.shape.0.address_phase_total_rounds()
    }

    fn degree(&self) -> usize {
        TWO_PHASE_DEGREE_BOUND
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(cycle_phase_intermediate_opening())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        final_output_expr(self.shape.1)
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
    fn formulas_evaluate_like_core_claims() {
        let dimensions = PrecommittedReductionDimensions::new(4, 3, true);
        let chunk_openings = [fr(17), fr(19)];
        let chunk_weights = [fr(23), fr(29)];
        let intermediate = fr(37);
        let zero = fr(0);

        let address = AddressPhase::new((dimensions, 2));
        let address_input = address.input_expression::<Fr>().evaluate(
            |id| {
                if *id == cycle_phase_intermediate_opening() {
                    intermediate
                } else {
                    zero
                }
            },
            |_| zero,
            |_| zero,
        );
        assert_eq!(address_input, intermediate);

        let output = address.output_expression::<Fr>().evaluate(
            |id| {
                (0..2)
                    .find(|&chunk| *id == final_bytecode_chunk_opening(chunk))
                    .map_or(zero, |chunk| chunk_openings[chunk])
            },
            |_| zero,
            |id| match *id {
                JoltDerivedId::BytecodeClaimReduction(
                    BytecodeClaimReductionPublic::ChunkOutputWeight(chunk),
                ) => chunk_weights[chunk],
                _ => zero,
            },
        );
        assert_eq!(
            output,
            chunk_weights[0] * chunk_openings[0] + chunk_weights[1] * chunk_openings[1]
        );
    }

    #[test]
    fn address_phase_exposes_expected_dependencies() {
        let dimensions = PrecommittedReductionDimensions::new(4, 3, true);
        let relation = AddressPhase::new((dimensions, 2));

        assert_eq!(AddressPhase::id(), JoltRelationId::BytecodeClaimReduction);
        assert_eq!(relation.rounds(), dimensions.address_phase_total_rounds());
        assert_eq!(relation.degree(), TWO_PHASE_DEGREE_BOUND);
    }
}
