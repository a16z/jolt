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

    fn spec(&self) -> JoltSumcheckSpec {
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
    use jolt_field::{Fr, FromPrimitiveInt};

    fn dimensions() -> TraceDimensions {
        TraceDimensions::new(5)
    }

    #[test]
    fn claim_reduction_evaluates_like_core_formula() {
        let relation = ClaimReduction::new(dimensions());

        let lookup_spartan = Fr::from_u64(3);
        let left_lookup_spartan = Fr::from_u64(5);
        let right_lookup_spartan = Fr::from_u64(7);
        let left_input_spartan = Fr::from_u64(11);
        let right_input_spartan = Fr::from_u64(13);
        let lookup_reduced = Fr::from_u64(17);
        let left_lookup_reduced = Fr::from_u64(19);
        let right_lookup_reduced = Fr::from_u64(23);
        let left_input_reduced = Fr::from_u64(29);
        let right_input_reduced = Fr::from_u64(31);
        let gamma = Fr::from_u64(37);
        let eq_spartan = Fr::from_u64(41);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == lookup_output_spartan() => lookup_spartan,
                id if id == left_lookup_operand_spartan() => left_lookup_spartan,
                id if id == right_lookup_operand_spartan() => right_lookup_spartan,
                id if id == left_instruction_input_spartan() => left_input_spartan,
                id if id == right_instruction_input_spartan() => right_input_spartan,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::InstructionClaimReduction(
                    InstructionClaimReductionChallenge::Gamma,
                ) => gamma,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |_| zero,
        );

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == lookup_output_reduced() => lookup_reduced,
                id if id == left_lookup_operand_reduced() => left_lookup_reduced,
                id if id == right_lookup_operand_reduced() => right_lookup_reduced,
                id if id == left_instruction_input_reduced() => left_input_reduced,
                id if id == right_instruction_input_reduced() => right_input_reduced,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::InstructionClaimReduction(
                    InstructionClaimReductionChallenge::Gamma,
                ) => gamma,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |id| match *id {
                JoltPublicId::InstructionClaimReduction(
                    InstructionClaimReductionPublic::EqSpartan,
                ) => eq_spartan,
                _ => zero,
            },
        );

        assert_eq!(
            input,
            lookup_spartan
                + gamma * left_lookup_spartan
                + gamma * gamma * right_lookup_spartan
                + gamma * gamma * gamma * left_input_spartan
                + gamma * gamma * gamma * gamma * right_input_spartan
        );
        assert_eq!(
            output,
            eq_spartan
                * (lookup_reduced
                    + gamma * left_lookup_reduced
                    + gamma * gamma * right_lookup_reduced
                    + gamma * gamma * gamma * left_input_reduced
                    + gamma * gamma * gamma * gamma * right_input_reduced)
        );
    }

    #[test]
    fn claim_reduction_exposes_expected_dependencies() {
        let relation = ClaimReduction::new(dimensions());

        assert_eq!(
            ClaimReduction::id(),
            JoltRelationId::InstructionClaimReduction
        );
        assert_eq!(relation.spec(), dimensions().sumcheck(2));
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
