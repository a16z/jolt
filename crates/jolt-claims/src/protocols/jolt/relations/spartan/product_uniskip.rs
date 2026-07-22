//! Spartan product univariate-skip symbolic sumcheck relation.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::dimensions::{
    PRODUCT_UNISKIP_DOMAIN_SIZE, PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
};
use crate::protocols::jolt::geometry::spartan::{
    product_outer_opening, product_should_branch_outer_opening, product_should_jump_outer_opening,
    product_uniskip_opening, product_uniskip_weight, SpartanProductDimensions,
};
use crate::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId,
};
use crate::{opening, InputClaims, OutputClaims, SumcheckDomain, SymbolicSumcheck};

/// Consumed product uni-skip inputs: the three Spartan-outer openings the first
/// round reduces (`product`, `should_branch`, `should_jump`), each reweighted by a
/// `UniskipLagrangeWeight`. The relation reads only their values (the input claim
/// is the pre-binding sum), so the input points are left empty. Generic over the
/// cell.
#[derive(Clone, Debug, InputClaims)]
pub struct ProductUniskipInputClaims<C> {
    #[opening(Product, from = SpartanOuter)]
    pub product: C,
    #[opening(ShouldBranch, from = SpartanOuter)]
    pub should_branch: C,
    #[opening(ShouldJump, from = SpartanOuter)]
    pub should_jump: C,
}

/// Produced product uni-skip opening (the single reduced univariate-skip value).
/// Generic over the opening cell (`F` for the serialized wire value, `Vec<F>` for
/// the derived opening point).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(SpartanProductVirtualization)]
pub struct ProductUniskipOutputClaims<C> {
    #[opening(UnivariateSkip)]
    pub uniskip: C,
}

/// The Spartan product univariate-skip sumcheck (first round). A standalone
/// centered-integer sumcheck whose reduced opening feeds the product remainder.
#[derive(Clone)]
pub struct ProductUniskip;

impl SymbolicSumcheck for ProductUniskip {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = SpartanProductDimensions;
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = ProductUniskipInputClaims<C>;
    type Outputs<C> = ProductUniskipOutputClaims<C>;

    fn new(_shape: SpartanProductDimensions) -> Self {
        Self
    }

    fn id() -> JoltRelationId {
        JoltRelationId::SpartanProductVirtualization
    }

    fn domain(&self) -> SumcheckDomain {
        SumcheckDomain::centered_integer(PRODUCT_UNISKIP_DOMAIN_SIZE)
    }

    fn rounds(&self) -> usize {
        1
    }

    fn degree(&self) -> usize {
        PRODUCT_UNISKIP_FIRST_ROUND_DEGREE
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
