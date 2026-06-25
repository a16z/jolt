use super::super::super::{JoltOpeningId, JoltRelationId, JoltVirtualPolynomial};

pub fn claim_reduction_input_openings() -> [JoltOpeningId; 3] {
    [
        rd_write_value_spartan(),
        rs1_value_spartan(),
        rs2_value_spartan(),
    ]
}

pub fn claim_reduction_output_openings() -> [JoltOpeningId; 3] {
    [
        rd_write_value_reduced(),
        rs1_value_reduced(),
        rs2_value_reduced(),
    ]
}

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

pub(crate) fn rd_write_value_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RdWriteValue,
        JoltRelationId::RegistersClaimReduction,
    )
}

pub(crate) fn rs1_value_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs1Value,
        JoltRelationId::RegistersClaimReduction,
    )
}

pub(crate) fn rs2_value_reduced() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs2Value,
        JoltRelationId::RegistersClaimReduction,
    )
}
