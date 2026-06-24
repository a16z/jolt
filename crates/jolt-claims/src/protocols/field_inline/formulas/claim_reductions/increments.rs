use jolt_field::RingCore;

use crate::{challenge, public};

use super::super::super::{
    FieldInlineChallengeId, FieldInlineCommittedPolynomial, FieldInlineExpr, FieldInlineOpeningId,
    FieldInlinePublicId, FieldInlineRelationId, FieldRegistersIncClaimReductionChallenge,
    FieldRegistersIncClaimReductionPublic,
};

pub fn claim_reduction_input_openings() -> [FieldInlineOpeningId; 2] {
    [field_rd_inc_read_write(), field_rd_inc_val_evaluation()]
}

pub fn claim_reduction_output_openings() -> [FieldInlineOpeningId; 1] {
    [field_rd_inc_reduced()]
}

pub fn field_rd_inc_read_write_opening() -> FieldInlineOpeningId {
    field_rd_inc_read_write()
}

pub fn field_rd_inc_val_evaluation_opening() -> FieldInlineOpeningId {
    field_rd_inc_val_evaluation()
}

pub fn field_rd_inc_reduced_opening() -> FieldInlineOpeningId {
    field_rd_inc_reduced()
}

pub(crate) fn inc_challenge<F>(id: FieldRegistersIncClaimReductionChallenge) -> FieldInlineExpr<F>
where
    F: RingCore,
{
    challenge(FieldInlineChallengeId::from(id))
}

pub(crate) fn inc_public<F>(id: FieldRegistersIncClaimReductionPublic) -> FieldInlineExpr<F>
where
    F: RingCore,
{
    public(FieldInlinePublicId::from(id))
}

pub(crate) fn field_rd_inc_read_write() -> FieldInlineOpeningId {
    FieldInlineOpeningId::committed(
        FieldInlineCommittedPolynomial::FieldRdInc,
        FieldInlineRelationId::FieldRegistersReadWriteChecking,
    )
}

pub(crate) fn field_rd_inc_val_evaluation() -> FieldInlineOpeningId {
    FieldInlineOpeningId::committed(
        FieldInlineCommittedPolynomial::FieldRdInc,
        FieldInlineRelationId::FieldRegistersValEvaluation,
    )
}

pub(crate) fn field_rd_inc_reduced() -> FieldInlineOpeningId {
    FieldInlineOpeningId::committed(
        FieldInlineCommittedPolynomial::FieldRdInc,
        FieldInlineRelationId::FieldRegistersIncClaimReduction,
    )
}
