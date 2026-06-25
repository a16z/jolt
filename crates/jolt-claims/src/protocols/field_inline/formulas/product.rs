use super::super::{FieldInlineOpeningId, FieldInlineRelationId, FieldInlineVirtualPolynomial};

pub fn field_product_input_openings() -> [FieldInlineOpeningId; 1] {
    [field_product_opening()]
}

pub fn field_product_output_openings() -> [FieldInlineOpeningId; 2] {
    [field_rs1_value_product(), field_rs2_value_product()]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldRegistersProductLane {
    Product,
    InverseProduct,
}

impl FieldRegistersProductLane {
    pub fn input_opening(self) -> FieldInlineOpeningId {
        match self {
            Self::Product => field_product_opening(),
            Self::InverseProduct => field_inv_product_opening(),
        }
    }

    pub fn factor_openings(self) -> [FieldInlineOpeningId; 2] {
        match self {
            Self::Product => [field_rs1_value_product(), field_rs2_value_product()],
            Self::InverseProduct => [field_rs1_value_product(), field_rd_value_product()],
        }
    }
}

pub const fn selected_product_lanes() -> [FieldRegistersProductLane; 2] {
    [
        FieldRegistersProductLane::Product,
        FieldRegistersProductLane::InverseProduct,
    ]
}

pub fn selected_product_uniskip_input_openings() -> [FieldInlineOpeningId; 2] {
    selected_product_lanes().map(FieldRegistersProductLane::input_opening)
}

pub fn selected_product_remainder_output_openings() -> [FieldInlineOpeningId; 3] {
    [
        field_rs1_value_product(),
        field_rs2_value_product(),
        field_rd_value_product(),
    ]
}

pub(crate) fn field_product_opening() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldProduct,
        FieldInlineRelationId::FieldRegistersProduct,
    )
}

pub(crate) fn field_inv_product_opening() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldInvProduct,
        FieldInlineRelationId::FieldRegistersProduct,
    )
}

pub(crate) fn field_rs1_value_product() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRs1Value,
        FieldInlineRelationId::FieldRegistersProduct,
    )
}

pub(crate) fn field_rs2_value_product() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRs2Value,
        FieldInlineRelationId::FieldRegistersProduct,
    )
}

pub(crate) fn field_rd_value_product() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRdValue,
        FieldInlineRelationId::FieldRegistersProduct,
    )
}
