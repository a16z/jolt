//! Spartan product univariate-skip symbolic sumcheck relation.

use jolt_field::RingCore;

use crate::protocols::jolt::geometry::spartan::{
    product_outer_opening, product_should_branch_outer_opening, product_should_jump_outer_opening,
    product_uniskip_opening, product_uniskip_weight, SpartanProductDimensions,
};
use crate::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId, JoltSumcheckSpec,
};
use crate::{opening, SymbolicSumcheck};

/// The Spartan product univariate-skip sumcheck (first round). Symbolic-only:
/// special-cased in the verifier's stage 2.
pub struct ProductUniskip {
    shape: SpartanProductDimensions,
}

impl SymbolicSumcheck for ProductUniskip {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = SpartanProductDimensions;

    fn new(shape: SpartanProductDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::SpartanProductVirtualization
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.uniskip_sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        product_uniskip_weight(0) * opening(product_outer_opening())
            + product_uniskip_weight(1) * opening(product_should_branch_outer_opening())
            + product_uniskip_weight(2) * opening(product_should_jump_outer_opening())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(product_uniskip_opening())
    }
}
