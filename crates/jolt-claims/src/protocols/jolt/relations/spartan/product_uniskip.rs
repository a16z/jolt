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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::dimensions::{
        PRODUCT_UNISKIP_DOMAIN_SIZE, PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
    };
    use crate::protocols::jolt::SpartanProductVirtualizationPublic;
    use jolt_field::{Fr, FromPrimitiveInt};

    /// The input claim is the Lagrange-reweighted sum of the three consumed
    /// Spartan-outer openings: `w0*product + w1*should_branch + w2*should_jump`.
    /// Distinct primes per source make any dropped, duplicated, or swapped term
    /// change the total.
    #[test]
    fn input_expression_evaluates_like_lagrange_weighted_sum() {
        let relation = ProductUniskip::new(SpartanProductDimensions::new(7));

        let product = Fr::from_u64(2);
        let should_branch = Fr::from_u64(3);
        let should_jump = Fr::from_u64(5);
        let weights = [Fr::from_u64(7), Fr::from_u64(11), Fr::from_u64(13)];
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == product_outer_opening() => product,
                id if id == product_should_branch_outer_opening() => should_branch,
                id if id == product_should_jump_outer_opening() => should_jump,
                _ => zero,
            },
            |_| zero,
            |id| match *id {
                JoltDerivedId::SpartanProductVirtualization(
                    SpartanProductVirtualizationPublic::UniskipLagrangeWeight(index),
                ) => weights[index],
                _ => zero,
            },
        );

        assert_eq!(
            input,
            weights[0] * product + weights[1] * should_branch + weights[2] * should_jump
        );
    }

    /// The output claim is the single reduced uni-skip opening passed through
    /// verbatim (coefficient one, no other sources), and the symbolically
    /// derived produced-opening set contains exactly that id.
    #[test]
    fn output_expression_is_the_uniskip_opening_verbatim() {
        let relation = ProductUniskip::new(SpartanProductDimensions::new(7));
        let uniskip_value = Fr::from_u64(59);

        let output = relation.output_expression::<Fr>().evaluate(
            |id| {
                assert_eq!(*id, product_uniskip_opening(), "unexpected opening {id:?}");
                uniskip_value
            },
            |id| unreachable!("output expression must not read challenge {id:?}"),
            |id| unreachable!("output expression must not read derived value {id:?}"),
        );
        assert_eq!(output, uniskip_value);

        assert_eq!(
            relation.expected_output_openings::<Fr>(),
            std::iter::once(product_uniskip_opening()).collect(),
        );
    }

    /// Pins the uni-skip sumcheck spec: a single round over the centered-integer
    /// domain, with the shared geometry constants for size and degree.
    #[test]
    fn sumcheck_spec_matches_uniskip_geometry_constants() {
        let relation = ProductUniskip::new(SpartanProductDimensions::new(7));
        assert_eq!(
            ProductUniskip::id(),
            JoltRelationId::SpartanProductVirtualization
        );
        assert_eq!(relation.rounds(), 1);
        assert_eq!(relation.degree(), PRODUCT_UNISKIP_FIRST_ROUND_DEGREE);
        assert_eq!(
            relation.domain(),
            SumcheckDomain::centered_integer(PRODUCT_UNISKIP_DOMAIN_SIZE),
        );
    }

    /// The derived `InputClaims` wiring: each consumed opening resolves under
    /// its Spartan-outer id in field-declaration order, and an id from a
    /// different relation resolves to `None`.
    #[test]
    fn input_claims_resolve_by_spartan_outer_ids_in_declaration_order() {
        let claims = ProductUniskipInputClaims {
            product: Fr::from_u64(2),
            should_branch: Fr::from_u64(3),
            should_jump: Fr::from_u64(5),
        };

        assert_eq!(
            claims.canonical_order(),
            vec![
                product_outer_opening(),
                product_should_branch_outer_opening(),
                product_should_jump_outer_opening(),
            ],
        );
        assert_eq!(
            claims.resolve_input(&product_outer_opening()),
            Some(Fr::from_u64(2)),
        );
        assert_eq!(
            claims.resolve_input(&product_should_branch_outer_opening()),
            Some(Fr::from_u64(3)),
        );
        assert_eq!(
            claims.resolve_input(&product_should_jump_outer_opening()),
            Some(Fr::from_u64(5)),
        );
        assert_eq!(claims.resolve_input(&product_uniskip_opening()), None);
    }
}
