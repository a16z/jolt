//! Instruction claim-reduction symbolic sumcheck relation.

use jolt_field::RingCore;

use crate::protocols::jolt::formulas::claim_reductions::instruction::{
    left_instruction_input_reduced, left_instruction_input_spartan, left_lookup_operand_reduced,
    left_lookup_operand_spartan, lookup_output_reduced, lookup_output_spartan, reduction_public,
    right_instruction_input_reduced, right_instruction_input_spartan, right_lookup_operand_reduced,
    right_lookup_operand_spartan, weighted_claims,
};
use crate::protocols::jolt::{
    InstructionClaimReductionPublic, JoltChallengeId, JoltExpr, JoltOpeningId, JoltPublicId,
    JoltRelationId, JoltSumcheckSpec, TraceDimensions,
};
use crate::SymbolicSumcheck;

/// Batches the Spartan-outer instruction-lookup openings (lookup output, left/
/// right lookup operands, left/right instruction inputs) by `gamma` and reduces
/// them to the instruction-claim-reduction openings weighted by `EqSpartan`.
pub struct ClaimReduction {
    shape: TraceDimensions,
}

impl SymbolicSumcheck for ClaimReduction {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type PublicId = JoltPublicId;
    type ChallengeId = JoltChallengeId;
    type Shape = TraceDimensions;

    fn new(shape: TraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::InstructionClaimReduction
    }

    fn sumcheck(&self) -> JoltSumcheckSpec {
        self.shape.sumcheck(2)
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        weighted_claims(
            lookup_output_spartan(),
            left_lookup_operand_spartan(),
            right_lookup_operand_spartan(),
            left_instruction_input_spartan(),
            right_instruction_input_spartan(),
        )
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        reduction_public(InstructionClaimReductionPublic::EqSpartan)
            * weighted_claims(
                lookup_output_reduced(),
                left_lookup_operand_reduced(),
                right_lookup_operand_reduced(),
                left_instruction_input_reduced(),
                right_instruction_input_reduced(),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::InstructionClaimReductionChallenge;
    use jolt_field::Fr;

    fn dimensions() -> TraceDimensions {
        TraceDimensions::new(5)
    }

    #[test]
    fn claim_reduction_exposes_expected_dependencies() {
        let relation = ClaimReduction::new(dimensions());

        assert_eq!(
            ClaimReduction::id(),
            JoltRelationId::InstructionClaimReduction
        );
        assert_eq!(relation.sumcheck(), dimensions().sumcheck(2));
        assert_eq!(
            relation.input_expression::<Fr>().required_openings(),
            vec![
                lookup_output_spartan(),
                left_lookup_operand_spartan(),
                right_lookup_operand_spartan(),
                left_instruction_input_spartan(),
                right_instruction_input_spartan(),
            ]
        );
        assert_eq!(
            relation.output_expression::<Fr>().required_openings(),
            vec![
                lookup_output_reduced(),
                left_lookup_operand_reduced(),
                right_lookup_operand_reduced(),
                left_instruction_input_reduced(),
                right_instruction_input_reduced(),
            ]
        );
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![JoltChallengeId::from(
                InstructionClaimReductionChallenge::Gamma
            )]
        );
        assert_eq!(
            relation.required_publics::<Fr>(),
            vec![JoltPublicId::from(
                InstructionClaimReductionPublic::EqSpartan
            )]
        );
    }
}
