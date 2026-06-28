use super::super::super::{JoltOpeningId, JoltRelationId, JoltVirtualPolynomial};

pub(crate) fn rd_write_value_spartan() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RdWriteValue,
        JoltRelationId::SpartanOuter,
    )
}

pub(crate) fn rs1_value_spartan() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs1Value,
        JoltRelationId::SpartanOuter,
    )
}

pub(crate) fn rs2_value_spartan() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs2Value,
        JoltRelationId::SpartanOuter,
    )
}

pub fn rd_write_value_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RdWriteValue,
        JoltRelationId::RegistersClaimReduction,
    )
}

pub fn rs1_value_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs1Value,
        JoltRelationId::RegistersClaimReduction,
    )
}

pub fn rs2_value_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs2Value,
        JoltRelationId::RegistersClaimReduction,
    )
}
