use jolt_field::RingCore;

use crate::{challenge, opening, public};

use super::super::super::{
    InstructionClaimReductionChallenge, InstructionClaimReductionPublic, JoltChallengeId, JoltExpr,
    JoltOpeningId, JoltPublicId, JoltRelationId, JoltVirtualPolynomial,
};

pub fn claim_reduction_output_openings() -> [JoltOpeningId; 5] {
    [
        lookup_output_reduced(),
        left_lookup_operand_reduced(),
        right_lookup_operand_reduced(),
        left_instruction_input_reduced(),
        right_instruction_input_reduced(),
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

pub(crate) fn weighted_claims<F>(
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

pub(crate) fn reduction_challenge<F>(id: InstructionClaimReductionChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

pub(crate) fn reduction_public<F>(id: InstructionClaimReductionPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

pub(crate) fn lookup_output_spartan() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LookupOutput,
        JoltRelationId::SpartanOuter,
    )
}

pub(crate) fn left_lookup_operand_spartan() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LeftLookupOperand,
        JoltRelationId::SpartanOuter,
    )
}

pub(crate) fn right_lookup_operand_spartan() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RightLookupOperand,
        JoltRelationId::SpartanOuter,
    )
}

pub(crate) fn left_instruction_input_spartan() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LeftInstructionInput,
        JoltRelationId::SpartanOuter,
    )
}

pub(crate) fn right_instruction_input_spartan() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RightInstructionInput,
        JoltRelationId::SpartanOuter,
    )
}

pub(crate) fn lookup_output_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LookupOutput,
        JoltRelationId::InstructionClaimReduction,
    )
}

pub(crate) fn left_lookup_operand_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LeftLookupOperand,
        JoltRelationId::InstructionClaimReduction,
    )
}

pub(crate) fn right_lookup_operand_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RightLookupOperand,
        JoltRelationId::InstructionClaimReduction,
    )
}

pub(crate) fn left_instruction_input_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LeftInstructionInput,
        JoltRelationId::InstructionClaimReduction,
    )
}

pub(crate) fn right_instruction_input_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RightInstructionInput,
        JoltRelationId::InstructionClaimReduction,
    )
}
