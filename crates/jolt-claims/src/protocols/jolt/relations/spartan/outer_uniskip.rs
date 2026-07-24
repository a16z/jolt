//! Spartan outer univariate-skip symbolic sumcheck relation.

use core::marker::PhantomData;

use jolt_field::{Field, RingCore};
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::dimensions::{
    OUTER_UNISKIP_DOMAIN_SIZE, OUTER_UNISKIP_FIRST_ROUND_DEGREE,
};
use crate::protocols::jolt::geometry::spartan::{outer_uniskip_opening, SpartanOuterDimensions};
use crate::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId,
};
use crate::{opening, InputClaims, OutputClaims, SumcheckDomain, SymbolicSumcheck};

/// The Spartan outer uni-skip consumes no openings (its input claim is the constant
/// zero), so this carries only the cell marker. Hand-implements [`InputClaims`]
/// since the derive requires at least one `#[opening]` field.
pub struct OuterUniskipInputClaims<C> {
    _cell: PhantomData<C>,
}

impl<C> Default for OuterUniskipInputClaims<C> {
    fn default() -> Self {
        Self { _cell: PhantomData }
    }
}

impl<F: Field> InputClaims<F> for OuterUniskipInputClaims<F> {
    fn canonical_order(&self) -> Vec<JoltOpeningId> {
        Vec::new()
    }

    fn resolve_input(&self, _id: &JoltOpeningId) -> Option<F> {
        None
    }
}

/// Produced Spartan outer univariate-skip opening (the single reduced
/// univariate-skip value). Generic over the opening cell (`F` for the serialized
/// wire value, `Vec<F>` for the derived opening point).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(SpartanOuter)]
pub struct OuterUniskipOutputClaims<C> {
    #[opening(UnivariateSkip)]
    pub univariate_skip: C,
}

/// The Spartan outer univariate-skip sumcheck (first round). The concrete uni-skip
/// verification is special-cased (centered-integer domain) in the verifier's stage
/// 1; this relation supplies the input/output claim algebra. The input claim is
/// zero (the first round has no consumed openings), so the inputs stay empty.
pub struct OuterUniskip;

impl SymbolicSumcheck for OuterUniskip {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = SpartanOuterDimensions;
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = OuterUniskipInputClaims<C>;
    type Outputs<C> = OuterUniskipOutputClaims<C>;

    fn new(_shape: SpartanOuterDimensions) -> Self {
        Self
    }

    fn id() -> JoltRelationId {
        JoltRelationId::SpartanOuter
    }

    fn domain(&self) -> SumcheckDomain {
        SumcheckDomain::centered_integer(OUTER_UNISKIP_DOMAIN_SIZE)
    }

    fn rounds(&self) -> usize {
        1
    }

    fn degree(&self) -> usize {
        OUTER_UNISKIP_FIRST_ROUND_DEGREE
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        JoltExpr::zero()
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(outer_uniskip_opening())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::JoltVirtualPolynomial;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn dimensions() -> SpartanOuterDimensions {
        match SpartanOuterDimensions::new(
            8,
            vec![
                JoltVirtualPolynomial::PC,
                JoltVirtualPolynomial::LookupOutput,
            ],
            true,
        ) {
            Some(dimensions) => dimensions,
            None => unreachable!("test Spartan outer dimensions should be valid"),
        }
    }

    /// The uni-skip's input claim is the constant zero: the expression carries
    /// no leaves at all, so evaluation never consults any resolver (resolvers
    /// here panic to prove it) and returns zero.
    #[test]
    fn input_expression_is_leafless_constant_zero() {
        let relation = OuterUniskip::new(dimensions());
        assert!(relation.input_expression::<Fr>().is_zero());

        let input = relation.input_expression::<Fr>().evaluate(
            |id| unreachable!("input expression must not read opening {id:?}"),
            |id| unreachable!("input expression must not read challenge {id:?}"),
            |id| unreachable!("input expression must not read derived value {id:?}"),
        );
        assert_eq!(input, Fr::from_u64(0));
    }

    /// The output claim is the single reduced uni-skip opening passed through
    /// verbatim (coefficient one, no other sources), and the symbolically
    /// derived produced-opening set contains exactly that id.
    #[test]
    fn output_expression_is_the_uniskip_opening_verbatim() {
        let relation = OuterUniskip::new(dimensions());
        let uniskip_value = Fr::from_u64(41);

        let output = relation.output_expression::<Fr>().evaluate(
            |id| {
                assert_eq!(*id, outer_uniskip_opening(), "unexpected opening {id:?}");
                uniskip_value
            },
            |id| unreachable!("output expression must not read challenge {id:?}"),
            |id| unreachable!("output expression must not read derived value {id:?}"),
        );
        assert_eq!(output, uniskip_value);

        assert_eq!(
            relation.expected_output_openings::<Fr>(),
            std::iter::once(outer_uniskip_opening()).collect(),
        );
    }

    /// Pins the uni-skip sumcheck spec: a single round over the centered-integer
    /// domain, with the shared geometry constants for size and degree.
    #[test]
    fn sumcheck_spec_matches_uniskip_geometry_constants() {
        let relation = OuterUniskip::new(dimensions());
        assert_eq!(OuterUniskip::id(), JoltRelationId::SpartanOuter);
        assert_eq!(relation.rounds(), 1);
        assert_eq!(relation.degree(), OUTER_UNISKIP_FIRST_ROUND_DEGREE);
        assert_eq!(
            relation.domain(),
            SumcheckDomain::centered_integer(OUTER_UNISKIP_DOMAIN_SIZE),
        );
    }

    /// The hand-written `InputClaims` impl (the derive requires at least one
    /// field) must present an empty consumed-claim surface: no canonical ids
    /// and no resolvable values, not even for this relation's own opening.
    #[test]
    fn input_claims_resolve_nothing() {
        let claims = OuterUniskipInputClaims::<Fr>::default();
        assert!(claims.canonical_order().is_empty());
        assert_eq!(claims.resolve_input(&outer_uniskip_opening()), None);
    }
}
