use super::super::{
    FieldInlineCommittedPolynomial, FieldInlineOpeningId, FieldInlineRelationId,
    FieldInlineVirtualPolynomial,
};

pub fn read_write_checking_input_openings() -> [FieldInlineOpeningId; 3] {
    [
        field_rd_value_claim(),
        field_rs1_value_claim(),
        field_rs2_value_claim(),
    ]
}

pub fn read_write_checking_output_openings() -> [FieldInlineOpeningId; 5] {
    [
        field_registers_val_read_write(),
        field_rs1_ra_read_write(),
        field_rs2_ra_read_write(),
        field_rd_wa_read_write(),
        field_rd_inc_read_write(),
    ]
}

pub fn val_evaluation_input_openings() -> [FieldInlineOpeningId; 1] {
    [field_registers_val_read_write()]
}

pub fn val_evaluation_output_openings() -> [FieldInlineOpeningId; 2] {
    [field_rd_inc_val_evaluation(), field_rd_wa_val_evaluation()]
}

pub(crate) fn field_rd_value_claim() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRdValue,
        FieldInlineRelationId::FieldRegistersClaimReduction,
    )
}

pub(crate) fn field_rs1_value_claim() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRs1Value,
        FieldInlineRelationId::FieldRegistersClaimReduction,
    )
}

pub(crate) fn field_rs2_value_claim() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRs2Value,
        FieldInlineRelationId::FieldRegistersClaimReduction,
    )
}

pub(crate) fn field_registers_val_read_write() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRegistersVal,
        FieldInlineRelationId::FieldRegistersReadWriteChecking,
    )
}

pub(crate) fn field_rs1_ra_read_write() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRs1Ra,
        FieldInlineRelationId::FieldRegistersReadWriteChecking,
    )
}

pub(crate) fn field_rs2_ra_read_write() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRs2Ra,
        FieldInlineRelationId::FieldRegistersReadWriteChecking,
    )
}

pub(crate) fn field_rd_wa_read_write() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRdWa,
        FieldInlineRelationId::FieldRegistersReadWriteChecking,
    )
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

pub(crate) fn field_rd_wa_val_evaluation() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRdWa,
        FieldInlineRelationId::FieldRegistersValEvaluation,
    )
}
