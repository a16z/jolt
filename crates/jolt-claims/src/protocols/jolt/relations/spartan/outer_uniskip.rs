//! Spartan outer univariate-skip symbolic sumcheck relation.

use core::marker::PhantomData;

use jolt_field::{Field, RingCore};
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::spartan::{outer_uniskip_opening, SpartanOuterDimensions};
use crate::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId, JoltSumcheckSpec,
};
use crate::{opening, InputClaims, OutputClaims, SymbolicSumcheck};

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

impl<F: Field> InputClaims<F> for OuterUniskipInputClaims<crate::OpeningClaim<F>> {
    fn resolve_input(&self, _id: &JoltOpeningId) -> Option<F> {
        None
    }
}

/// Produced Spartan outer univariate-skip opening (the single reduced
/// univariate-skip value). Generic over the cell (`F` on the wire / serialized
/// proof form, `OpeningClaim<F>` on the clear path).
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
pub struct OuterUniskip {
    shape: SpartanOuterDimensions,
}

impl SymbolicSumcheck for OuterUniskip {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = SpartanOuterDimensions;
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = OuterUniskipInputClaims<C>;
    type Outputs<C> = OuterUniskipOutputClaims<C>;

    fn new(shape: SpartanOuterDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::SpartanOuter
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.uniskip_sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        JoltExpr::zero()
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(outer_uniskip_opening())
    }
}
