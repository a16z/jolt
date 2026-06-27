//! Spartan outer univariate-skip symbolic sumcheck relation.

use jolt_field::RingCore;

use crate::protocols::jolt::geometry::spartan::{outer_uniskip_opening, SpartanOuterDimensions};
use crate::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId, JoltSumcheckSpec,
};
use crate::{opening, SymbolicSumcheck};

/// The Spartan outer univariate-skip sumcheck (first round). Symbolic-only: the
/// concrete uni-skip verification is special-cased in the verifier's stage 1.
pub struct OuterUniskip {
    shape: SpartanOuterDimensions,
}

impl SymbolicSumcheck for OuterUniskip {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = SpartanOuterDimensions;

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
