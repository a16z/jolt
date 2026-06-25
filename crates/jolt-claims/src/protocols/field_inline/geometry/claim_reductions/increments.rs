use super::super::super::{
    FieldInlineCommittedPolynomial, FieldInlineOpeningId, FieldInlineRelationId,
};

pub fn claim_reduction_input_openings() -> [FieldInlineOpeningId; 2] {
    [field_rd_inc_read_write(), field_rd_inc_val_evaluation()]
}

pub fn claim_reduction_output_openings() -> [FieldInlineOpeningId; 1] {
    [field_rd_inc_reduced()]
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
