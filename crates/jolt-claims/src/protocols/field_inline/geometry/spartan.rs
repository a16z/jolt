use super::super::{FieldInlineOpeningId, FieldInlineRelationId, FieldInlineVirtualPolynomial};

pub const FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS: [FieldInlineVirtualPolynomial; 5] = [
    FieldInlineVirtualPolynomial::FieldRs1Value,
    FieldInlineVirtualPolynomial::FieldRs2Value,
    FieldInlineVirtualPolynomial::FieldRdValue,
    FieldInlineVirtualPolynomial::FieldProduct,
    FieldInlineVirtualPolynomial::FieldInvProduct,
];

pub const FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUT_COUNT: usize =
    FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS.len();

pub fn outer_opening(polynomial: FieldInlineVirtualPolynomial) -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        polynomial,
        FieldInlineRelationId::FieldRegistersSpartanOuter,
    )
}

pub fn outer_output_openings() -> [FieldInlineOpeningId; FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUT_COUNT]
{
    [
        outer_opening(FieldInlineVirtualPolynomial::FieldRs1Value),
        outer_opening(FieldInlineVirtualPolynomial::FieldRs2Value),
        outer_opening(FieldInlineVirtualPolynomial::FieldRdValue),
        outer_opening(FieldInlineVirtualPolynomial::FieldProduct),
        outer_opening(FieldInlineVirtualPolynomial::FieldInvProduct),
    ]
}
