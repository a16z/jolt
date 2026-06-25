//! Two-phase committed-bytecode claim-reduction symbolic relations.

use jolt_field::RingCore;

use crate::protocols::jolt::geometry::claim_reductions::bytecode::{
    assert_valid_chunk_count, bytecode_val_stage_opening, cycle_phase_intermediate_opening,
    final_output_expr, NUM_BYTECODE_VAL_STAGES,
};
use crate::protocols::jolt::{
    BytecodeClaimReductionChallenge, JoltChallengeId, JoltExpr, JoltOpeningId, JoltPublicId,
    JoltRelationId, JoltSumcheckSpec, PrecommittedReductionDimensions,
};
use crate::{challenge, opening, SymbolicSumcheck};

/// `(two-phase dimensions, chunk count)` shape shared by the committed-bytecode
/// cycle- and address-phase reductions.
pub type BytecodeReductionShape = (PrecommittedReductionDimensions, usize);

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
    type PublicId = JoltPublicId;
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

/// Address phase of the committed-bytecode reduction: reduces the cycle-phase
/// intermediate opening to the committed `BytecodeChunk(i)` openings weighted by
/// `ChunkOutputWeight`.
pub struct AddressPhase {
    shape: BytecodeReductionShape,
}

impl SymbolicSumcheck for AddressPhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type PublicId = JoltPublicId;
    type ChallengeId = JoltChallengeId;
    type Shape = BytecodeReductionShape;

    fn new(shape: BytecodeReductionShape) -> Self {
        assert_valid_chunk_count(shape.1);
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::BytecodeClaimReduction
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.0.address_sumcheck()
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
    use crate::protocols::jolt::BytecodeClaimReductionPublic;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    #[test]
    fn formulas_evaluate_like_core_claims() {
        let dimensions = PrecommittedReductionDimensions::new(4, 3, true);
        let eta = fr(31);
        let stage_claims = [fr(3), fr(5), fr(7), fr(11), fr(13)];
        let chunk_openings = [fr(17), fr(19)];
        let chunk_weights = [fr(23), fr(29)];
        let intermediate = fr(37);
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
                JoltPublicId::BytecodeClaimReduction(
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
        assert!(relation.required_publics::<Fr>().is_empty());
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
            relation.required_publics::<Fr>(),
            vec![
                JoltPublicId::from(BytecodeClaimReductionPublic::ChunkOutputWeight(0)),
                JoltPublicId::from(BytecodeClaimReductionPublic::ChunkOutputWeight(1)),
            ]
        );
    }

    #[test]
    fn address_phase_exposes_expected_dependencies() {
        let dimensions = PrecommittedReductionDimensions::new(4, 3, true);
        let relation = AddressPhase::new((dimensions, 2));

        assert_eq!(AddressPhase::id(), JoltRelationId::BytecodeClaimReduction);
        assert_eq!(relation.spec(), dimensions.address_sumcheck());
        assert_eq!(
            relation.input_expression::<Fr>().required_openings(),
            vec![cycle_phase_intermediate_opening()]
        );
        assert_eq!(
            relation.output_expression::<Fr>().required_openings(),
            vec![
                final_bytecode_chunk_opening(0),
                final_bytecode_chunk_opening(1),
            ]
        );
        assert_eq!(
            relation.required_publics::<Fr>(),
            vec![
                JoltPublicId::from(BytecodeClaimReductionPublic::ChunkOutputWeight(0)),
                JoltPublicId::from(BytecodeClaimReductionPublic::ChunkOutputWeight(1)),
            ]
        );
    }
}
