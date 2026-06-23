use jolt_field::RingCore;

use crate::{challenge, opening, public};

use super::super::super::{
    InstructionClaimReductionChallenge, InstructionClaimReductionPublic, JoltChallengeId, JoltExpr,
    JoltOpeningId, JoltPublicId, JoltRelationClaims, JoltRelationId, JoltVirtualPolynomial,
};
use super::super::dimensions::TraceDimensions;

pub fn claim_reduction<F>(dimensions: TraceDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    let input = weighted_claims(
        lookup_output_spartan(),
        left_lookup_operand_spartan(),
        right_lookup_operand_spartan(),
        left_instruction_input_spartan(),
        right_instruction_input_spartan(),
    );

    let output = reduction_public(InstructionClaimReductionPublic::EqSpartan)
        * weighted_claims(
            lookup_output_reduced(),
            left_lookup_operand_reduced(),
            right_lookup_operand_reduced(),
            left_instruction_input_reduced(),
            right_instruction_input_reduced(),
        );

    JoltRelationClaims::new(
        JoltRelationId::InstructionClaimReduction,
        dimensions.sumcheck(2),
        input,
        output,
    )
}

pub fn claim_reduction_output_openings() -> [JoltOpeningId; 5] {
    [
        lookup_output_reduced(),
        left_lookup_operand_reduced(),
        right_lookup_operand_reduced(),
        left_instruction_input_reduced(),
        right_instruction_input_reduced(),
    ]
}

pub fn stage2_claim_reduction_output_openings() -> [JoltOpeningId; 2] {
    [
        left_lookup_operand_reduced(),
        right_lookup_operand_reduced(),
    ]
}

pub fn claim_reduction_input_openings() -> [JoltOpeningId; 5] {
    [
        lookup_output_spartan(),
        left_lookup_operand_spartan(),
        right_lookup_operand_spartan(),
        left_instruction_input_spartan(),
        right_instruction_input_spartan(),
    ]
}

fn weighted_claims<F>(
    lookup_output: JoltOpeningId,
    left_lookup_operand: JoltOpeningId,
    right_lookup_operand: JoltOpeningId,
    left_instruction_input: JoltOpeningId,
    right_instruction_input: JoltOpeningId,
) -> JoltExpr<F>
where
    F: RingCore,
{
    let gamma = reduction_challenge(InstructionClaimReductionChallenge::Gamma);

    opening(lookup_output)
        + gamma.clone() * opening(left_lookup_operand)
        + gamma.clone().pow(2) * opening(right_lookup_operand)
        + gamma.clone().pow(3) * opening(left_instruction_input)
        + gamma.pow(4) * opening(right_instruction_input)
}

fn reduction_challenge<F>(id: InstructionClaimReductionChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

fn reduction_public<F>(id: InstructionClaimReductionPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

fn lookup_output_spartan() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LookupOutput,
        JoltRelationId::SpartanOuter,
    )
}

fn left_lookup_operand_spartan() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LeftLookupOperand,
        JoltRelationId::SpartanOuter,
    )
}

fn right_lookup_operand_spartan() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RightLookupOperand,
        JoltRelationId::SpartanOuter,
    )
}

fn left_instruction_input_spartan() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LeftInstructionInput,
        JoltRelationId::SpartanOuter,
    )
}

fn right_instruction_input_spartan() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RightInstructionInput,
        JoltRelationId::SpartanOuter,
    )
}

fn lookup_output_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LookupOutput,
        JoltRelationId::InstructionClaimReduction,
    )
}

fn left_lookup_operand_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LeftLookupOperand,
        JoltRelationId::InstructionClaimReduction,
    )
}

fn right_lookup_operand_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RightLookupOperand,
        JoltRelationId::InstructionClaimReduction,
    )
}

fn left_instruction_input_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LeftInstructionInput,
        JoltRelationId::InstructionClaimReduction,
    )
}

fn right_instruction_input_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RightInstructionInput,
        JoltRelationId::InstructionClaimReduction,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn dimensions() -> TraceDimensions {
        TraceDimensions::new(5)
    }

    #[test]
    fn claim_reduction_exposes_expected_dependencies() {
        let claims = claim_reduction::<Fr>(dimensions());

        assert_eq!(claims.id, JoltRelationId::InstructionClaimReduction);
        assert_eq!(claims.sumcheck, dimensions().sumcheck(2));
        assert_eq!(
            claims.input.required_openings,
            claim_reduction_input_openings().to_vec()
        );
        assert_eq!(
            claims.output.required_openings,
            claim_reduction_output_openings().to_vec()
        );
        assert_eq!(
            claims.input.required_challenges,
            vec![JoltChallengeId::from(
                InstructionClaimReductionChallenge::Gamma
            )]
        );
        assert_eq!(
            claims.output.required_challenges,
            vec![JoltChallengeId::from(
                InstructionClaimReductionChallenge::Gamma
            )]
        );
        assert_eq!(
            claims.required_challenges(),
            vec![JoltChallengeId::from(
                InstructionClaimReductionChallenge::Gamma
            )]
        );
        assert_eq!(
            claims.required_publics(),
            vec![JoltPublicId::from(
                InstructionClaimReductionPublic::EqSpartan
            )]
        );
        assert_eq!(claims.num_challenges(), 1);
    }

    #[test]
    fn stage2_claim_reduction_openings_are_reduced_lookup_operands() {
        let output_openings = claim_reduction_output_openings();

        assert_eq!(
            stage2_claim_reduction_output_openings(),
            [output_openings[1], output_openings[2]]
        );
    }

    #[test]
    fn claim_reduction_evaluates_like_core_formula() {
        let claims = claim_reduction::<Fr>(dimensions());

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

        let input = claims.input.expression().evaluate(
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

        let output = claims.output.expression().evaluate(
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
}
