use jolt_field::Ring;
use jolt_riscv::CircuitFlags;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::dimensions::{
    PRODUCT_UNISKIP_DOMAIN_SIZE, PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
};
use crate::protocols::jolt::geometry::spartan::{
    product_halt_lane_output, product_outer_opening, product_should_branch_outer_opening,
    product_uniskip_opening, product_uniskip_weight, SpartanProductDimensions,
};
use crate::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId,
};
use crate::{opening, InputClaims, OutputClaims, SumcheckDomain, SymbolicSumcheck};

/// Consumed product uni-skip inputs: the four Spartan-outer openings the first
/// round reduces. Lanes 1 and 2 are `product` and `should_branch`; lane 3 is the
/// fold of `halt` and `halt_panic` with the claimed panic bit
/// (`PRODUCT_HALT_LANE_OUTPUT`). Each lane is reweighted by a
/// `UniskipLagrangeWeight`. The relation reads only their values (the input
/// claim is the pre-binding sum), so the input points are left empty. Generic
/// over the cell.
#[derive(Clone, Debug, InputClaims)]
pub struct ProductUniskipInputClaims<C> {
    #[opening(Product, from = SpartanOuter)]
    pub product: C,
    #[opening(ShouldBranch, from = SpartanOuter)]
    pub should_branch: C,
    #[opening(OpFlags(CircuitFlags::Halt), from = SpartanOuter)]
    pub halt: C,
    #[opening(OpFlags(CircuitFlags::HaltPanic), from = SpartanOuter)]
    pub halt_panic: C,
}

/// Produced product uni-skip opening (the single reduced univariate-skip value).
/// Generic over the opening cell (`F` for the serialized wire value, `Vec<F>` for
/// the derived opening point).
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
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

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        product_uniskip_weight(0) * opening(product_outer_opening())
            + product_uniskip_weight(1) * opening(product_should_branch_outer_opening())
            + product_uniskip_weight(2) * product_halt_lane_output()
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        opening(product_uniskip_opening())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::spartan::outer_opening;
    use crate::protocols::jolt::JoltVirtualPolynomial;
    use jolt_field::{Fr, Ring};

    fn flag_opening(flag: CircuitFlags) -> JoltOpeningId {
        outer_opening(JoltVirtualPolynomial::OpFlags(flag))
    }

    #[test]
    fn input_claims_resolve_by_spartan_outer_ids_in_declaration_order() {
        let claims = ProductUniskipInputClaims {
            product: Fr::from_u64(2),
            should_branch: Fr::from_u64(3),
            halt: Fr::from_u64(7),
            halt_panic: Fr::from_u64(11),
        };
        let expected = [
            (product_outer_opening(), 2),
            (product_should_branch_outer_opening(), 3),
            (flag_opening(CircuitFlags::Halt), 7),
            (flag_opening(CircuitFlags::HaltPanic), 11),
        ];

        assert_eq!(
            claims.canonical_order(),
            expected.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        );
        for (id, value) in expected {
            assert_eq!(claims.resolve_input(&id), Some(Fr::from_u64(value)));
        }
        assert_eq!(claims.resolve_input(&product_uniskip_opening()), None);
    }
}
