use jolt_field::RingCore;

use crate::{challenge, public};

use super::super::super::{
    FieldInlineChallengeId, FieldInlineExpr, FieldInlineOpeningId, FieldInlinePublicId,
    FieldInlineRelationId, FieldInlineVirtualPolynomial, FieldRegistersClaimReductionChallenge,
    FieldRegistersClaimReductionPublic,
};

pub fn claim_reduction_input_openings() -> [FieldInlineOpeningId; 3] {
    [
        field_rd_value_spartan(),
        field_rs1_value_spartan(),
        field_rs2_value_spartan(),
    ]
}

pub fn claim_reduction_output_openings() -> [FieldInlineOpeningId; 3] {
    [
        field_rd_value_reduced(),
        field_rs1_value_reduced(),
        field_rs2_value_reduced(),
    ]
}

pub(crate) fn reduction_challenge<F>(
    id: FieldRegistersClaimReductionChallenge,
) -> FieldInlineExpr<F>
where
    F: RingCore,
{
    challenge(FieldInlineChallengeId::from(id))
}

pub(crate) fn reduction_public<F>(id: FieldRegistersClaimReductionPublic) -> FieldInlineExpr<F>
where
    F: RingCore,
{
    public(FieldInlinePublicId::from(id))
}

pub(crate) fn field_rd_value_spartan() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRdValue,
        FieldInlineRelationId::FieldRegistersSpartanOuter,
    )
}

pub(crate) fn field_rs1_value_spartan() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRs1Value,
        FieldInlineRelationId::FieldRegistersSpartanOuter,
    )
}

pub(crate) fn field_rs2_value_spartan() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRs2Value,
        FieldInlineRelationId::FieldRegistersSpartanOuter,
    )
}

pub(crate) fn field_rd_value_reduced() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRdValue,
        FieldInlineRelationId::FieldRegistersClaimReduction,
    )
}

pub(crate) fn field_rs1_value_reduced() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRs1Value,
        FieldInlineRelationId::FieldRegistersClaimReduction,
    )
}

pub(crate) fn field_rs2_value_reduced() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRs2Value,
        FieldInlineRelationId::FieldRegistersClaimReduction,
    )
}
