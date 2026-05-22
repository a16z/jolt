use jolt_field::RingCore;

use crate::opening;

use super::super::{
    FieldInlineOpeningId, FieldInlineRelationClaims, FieldInlineRelationId,
    FieldInlineVirtualPolynomial,
};
use super::dimensions::{FieldInlineSumcheckSpec, FieldRegistersTraceDimensions};

pub const fn field_product_sumcheck(
    dimensions: FieldRegistersTraceDimensions,
) -> FieldInlineSumcheckSpec {
    dimensions.sumcheck(2)
}

pub fn field_product<F>(dimensions: FieldRegistersTraceDimensions) -> FieldInlineRelationClaims<F>
where
    F: RingCore,
{
    FieldInlineRelationClaims::new(
        FieldInlineRelationId::FieldRegistersProduct,
        field_product_sumcheck(dimensions),
        opening(field_product_opening()),
        opening(field_rs1_value_product()) * opening(field_rs2_value_product()),
    )
}

pub fn field_product_input_openings() -> [FieldInlineOpeningId; 1] {
    [field_product_opening()]
}

pub fn field_product_output_openings() -> [FieldInlineOpeningId; 2] {
    [field_rs1_value_product(), field_rs2_value_product()]
}

pub fn selected_product_uniskip_input_openings() -> [FieldInlineOpeningId; 2] {
    [field_product_opening(), field_inv_product_opening()]
}

pub fn selected_product_remainder_output_openings() -> [FieldInlineOpeningId; 3] {
    [
        field_rs1_value_product(),
        field_rs2_value_product(),
        field_rd_value_product(),
    ]
}

fn field_product_opening() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldProduct,
        FieldInlineRelationId::FieldRegistersProduct,
    )
}

fn field_inv_product_opening() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldInvProduct,
        FieldInlineRelationId::FieldRegistersProduct,
    )
}

fn field_rs1_value_product() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRs1Value,
        FieldInlineRelationId::FieldRegistersProduct,
    )
}

fn field_rs2_value_product() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRs2Value,
        FieldInlineRelationId::FieldRegistersProduct,
    )
}

fn field_rd_value_product() -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        FieldInlineVirtualPolynomial::FieldRdValue,
        FieldInlineRelationId::FieldRegistersProduct,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn dimensions() -> FieldRegistersTraceDimensions {
        FieldRegistersTraceDimensions::new(5)
    }

    #[test]
    fn field_product_claims_expose_expected_dependencies() {
        let claims = field_product::<Fr>(dimensions());

        assert_eq!(claims.id, FieldInlineRelationId::FieldRegistersProduct);
        assert_eq!(claims.sumcheck, field_product_sumcheck(dimensions()));
        assert_eq!(
            claims.input.required_openings,
            field_product_input_openings().to_vec()
        );
        assert_eq!(
            claims.output.required_openings,
            field_product_output_openings().to_vec()
        );
        assert!(claims.required_challenges().is_empty());
        assert!(claims.required_publics().is_empty());
        assert_eq!(claims.num_challenges(), 0);
        assert_eq!(
            selected_product_uniskip_input_openings(),
            [field_product_opening(), field_inv_product_opening()]
        );
        assert_eq!(
            selected_product_remainder_output_openings(),
            [
                field_rs1_value_product(),
                field_rs2_value_product(),
                field_rd_value_product(),
            ]
        );
    }

    #[test]
    fn field_product_claims_evaluate_native_field_product_relation() {
        let claims = field_product::<Fr>(dimensions());

        let product = Fr::from_u64(35);
        let rs1 = Fr::from_u64(5);
        let rs2 = Fr::from_u64(7);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression().evaluate(
            |id| match *id {
                id if id == field_product_opening() => product,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );

        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == field_rs1_value_product() => rs1,
                id if id == field_rs2_value_product() => rs2,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );

        assert_eq!(input, product);
        assert_eq!(output, rs1 * rs2);
        assert_eq!(input, output);
    }
}
