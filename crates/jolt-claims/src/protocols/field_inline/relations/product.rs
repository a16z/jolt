//! field_inline native product symbolic sumcheck relation.

use jolt_field::Ring;
use serde::{Deserialize, Serialize};

use crate::opening;
use crate::protocols::field_inline::geometry::product::{
    field_product_opening, field_rs1_value_product, field_rs2_value_product,
};
use crate::protocols::field_inline::{
    FieldInlineChallengeId, FieldInlineDerivedId, FieldInlineExpr, FieldInlineOpeningId,
    FieldInlineRelationId, FieldRegistersTraceDimensions,
};
use crate::SymbolicSumcheck;
use crate::{InputClaims, OutputClaims};

/// Produced field-product openings: the three factor openings the selected FR
/// lanes reference at the shared product-remainder point (`FieldRdValue` is the
/// `FieldInvProduct` lane's right factor). Field declaration order is the
/// canonical Fiat-Shamir order and mirrors
/// `geometry::product::selected_product_remainder_output_openings()`.
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[protocol(field_inline)]
#[relation(FieldRegistersProduct)]
pub struct FieldRegistersProductOutputClaims<C> {
    #[opening(FieldRs1Value)]
    pub rs1_value: C,
    #[opening(FieldRs2Value)]
    pub rs2_value: C,
    #[opening(FieldRdValue)]
    pub rd_value: C,
}

/// Consumed field-product input: the `FieldProduct` lane claim entering the
/// product virtualization.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
#[protocol(field_inline)]
pub struct FieldRegistersProductInputClaims<C> {
    #[opening(FieldProduct, from = FieldRegistersProduct)]
    pub field_product: C,
}

/// The native field-register product sumcheck: equates the `FieldProduct` opening
/// to `FieldRs1Value * FieldRs2Value`.
pub struct FieldProduct {
    shape: FieldRegistersTraceDimensions,
}

impl SymbolicSumcheck for FieldProduct {
    type RelationId = FieldInlineRelationId;
    type OpeningId = FieldInlineOpeningId;
    type DerivedId = FieldInlineDerivedId;
    type ChallengeId = FieldInlineChallengeId;
    type Shape = FieldRegistersTraceDimensions;
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = FieldRegistersProductInputClaims<C>;
    type Outputs<C> = FieldRegistersProductOutputClaims<C>;

    fn new(shape: FieldRegistersTraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> FieldInlineRelationId {
        FieldInlineRelationId::FieldRegistersProduct
    }

    fn rounds(&self) -> usize {
        self.shape.log_t()
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<F: Ring>(&self) -> FieldInlineExpr<F> {
        opening(field_product_opening())
    }

    fn output_expression<F: Ring>(&self) -> FieldInlineExpr<F> {
        opening(field_rs1_value_product()) * opening(field_rs2_value_product())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::field_inline::geometry::product::{
        field_inv_product_opening, field_rd_value_product, selected_product_lanes,
        selected_product_remainder_output_openings, selected_product_uniskip_input_openings,
        FieldRegistersProductLane,
    };
    use jolt_field::{Fr, Ring};

    fn dimensions() -> FieldRegistersTraceDimensions {
        FieldRegistersTraceDimensions::new(5)
    }

    #[test]
    fn claim_struct_field_order_matches_geometry_opening_order() {
        use crate::protocols::field_inline::geometry::product::field_product_input_openings;

        let value = Fr::from_u64(1);

        let outputs = FieldRegistersProductOutputClaims::<Fr> {
            rs1_value: value,
            rs2_value: value,
            rd_value: value,
        };
        assert_eq!(
            outputs.canonical_order(),
            selected_product_remainder_output_openings()
        );

        let inputs = FieldRegistersProductInputClaims::<Fr> {
            field_product: value,
        };
        assert_eq!(inputs.canonical_order(), field_product_input_openings());
    }

    #[test]
    fn field_product_claims_expose_expected_dependencies() {
        let relation = FieldProduct::new(dimensions());

        assert_eq!(
            FieldProduct::id(),
            FieldInlineRelationId::FieldRegistersProduct
        );
        assert_eq!(relation.rounds(), dimensions().log_t());
        assert_eq!(relation.degree(), 2);
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
