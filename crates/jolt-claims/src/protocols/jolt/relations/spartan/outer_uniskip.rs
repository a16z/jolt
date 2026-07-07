//! Spartan outer univariate-skip symbolic sumcheck relation.

use core::marker::PhantomData;

use jolt_field::{Field, RingCore};
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::dimensions::{
    OUTER_UNISKIP_DOMAIN_SIZE, OUTER_UNISKIP_FIRST_ROUND_DEGREE,
};
use crate::protocols::jolt::geometry::spartan::{outer_uniskip_opening, SpartanOuterDimensions};
use crate::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId, JoltSumcheckDomain,
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
        JoltSumcheckDomain::centered_integer(OUTER_UNISKIP_DOMAIN_SIZE)
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
