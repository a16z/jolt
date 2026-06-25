//! field_inline native product symbolic sumcheck relation.

use jolt_field::RingCore;

use crate::opening;
use crate::protocols::field_inline::geometry::product::{
    field_product_opening, field_rs1_value_product, field_rs2_value_product,
};
use crate::protocols::field_inline::{
    FieldInlineChallengeId, FieldInlineExpr, FieldInlineOpeningId, FieldInlinePublicId,
    FieldInlineRelationId, FieldInlineSumcheckSpec, FieldRegistersTraceDimensions,
};
use crate::SymbolicSumcheck;

/// The native field-register product sumcheck: equates the `FieldProduct` opening
/// to `FieldRs1Value * FieldRs2Value`.
pub struct FieldProduct {
    shape: FieldRegistersTraceDimensions,
}

impl SymbolicSumcheck for FieldProduct {
    type RelationId = FieldInlineRelationId;
    type OpeningId = FieldInlineOpeningId;
    type PublicId = FieldInlinePublicId;
    type ChallengeId = FieldInlineChallengeId;
    type Shape = FieldRegistersTraceDimensions;

    fn new(shape: FieldRegistersTraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> FieldInlineRelationId {
        FieldInlineRelationId::FieldRegistersProduct
    }

    fn spec(&self) -> FieldInlineSumcheckSpec {
        self.shape.sumcheck(2)
    }

    fn input_expression<F: RingCore>(&self) -> FieldInlineExpr<F> {
        opening(field_product_opening())
    }

    fn output_expression<F: RingCore>(&self) -> FieldInlineExpr<F> {
        opening(field_rs1_value_product()) * opening(field_rs2_value_product())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::field_inline::geometry::product::{
        field_inv_product_opening, field_product_input_openings, field_product_output_openings,
        field_rd_value_product, selected_product_lanes, selected_product_remainder_output_openings,
        selected_product_uniskip_input_openings, FieldRegistersProductLane,
    };
    use jolt_field::{Fr, FromPrimitiveInt};

    fn dimensions() -> FieldRegistersTraceDimensions {
        FieldRegistersTraceDimensions::new(5)
    }

    #[test]
    fn field_product_claims_expose_expected_dependencies() {
        let relation = FieldProduct::new(dimensions());

        assert_eq!(
            FieldProduct::id(),
            FieldInlineRelationId::FieldRegistersProduct
        );
        assert_eq!(relation.spec(), dimensions().sumcheck(2));
        assert_eq!(
            relation.input_expression::<Fr>().required_openings(),
            field_product_input_openings().to_vec()
        );
        assert_eq!(
            relation.output_expression::<Fr>().required_openings(),
            field_product_output_openings().to_vec()
        );
        assert!(relation.required_challenges::<Fr>().is_empty());
        assert!(relation.required_publics::<Fr>().is_empty());
        assert_eq!(
            selected_product_uniskip_input_openings(),
            [field_product_opening(), field_inv_product_opening()]
        );
        assert_eq!(
            selected_product_lanes().map(FieldRegistersProductLane::factor_openings),
            [
                [field_rs1_value_product(), field_rs2_value_product()],
                [field_rs1_value_product(), field_rd_value_product()],
            ]
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
        let relation = FieldProduct::new(dimensions());

        let product = Fr::from_u64(35);
        let rs1 = Fr::from_u64(5);
        let rs2 = Fr::from_u64(7);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == field_product_opening() => product,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );

        let output = relation.output_expression::<Fr>().evaluate(
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
