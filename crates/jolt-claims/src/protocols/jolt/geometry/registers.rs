use super::super::{JoltCommittedPolynomial, JoltOpeningId, JoltRelationId, JoltVirtualPolynomial};

pub(crate) fn rd_write_value_claim() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RdWriteValue,
        JoltRelationId::RegistersClaimReduction,
    )
}

pub(crate) fn rs1_value_claim() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs1Value,
        JoltRelationId::RegistersClaimReduction,
    )
}

pub(crate) fn rs2_value_claim() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs2Value,
        JoltRelationId::RegistersClaimReduction,
    )
}

pub fn registers_val_read_write() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RegistersVal,
        JoltRelationId::RegistersReadWriteChecking,
    )
}

pub fn rs1_ra_read_write() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs1Ra,
        JoltRelationId::RegistersReadWriteChecking,
    )
}

pub fn rs2_ra_read_write() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs2Ra,
        JoltRelationId::RegistersReadWriteChecking,
    )
}

pub fn rd_wa_read_write() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RdWa,
        JoltRelationId::RegistersReadWriteChecking,
    )
}

pub fn rd_inc_read_write() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RdInc,
        JoltRelationId::RegistersReadWriteChecking,
    )
}

pub fn rd_inc_val_evaluation() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RdInc,
        JoltRelationId::RegistersValEvaluation,
    )
}

pub fn rd_wa_val_evaluation() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RdWa,
        JoltRelationId::RegistersValEvaluation,
    )
}
