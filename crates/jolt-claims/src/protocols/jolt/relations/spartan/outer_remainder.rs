//! Spartan outer remainder symbolic sumcheck relation.

use jolt_field::RingCore;

use crate::protocols::jolt::geometry::spartan::{
    outer_opening, outer_uniskip_opening, SpartanOuterDimensions,
};
use crate::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId, JoltSumcheckSpec,
    SpartanOuterPublic,
};
use crate::{derived, opening, SymbolicSumcheck};

/// The Spartan outer remainder sumcheck: the quadratic R1CS form over the outer
/// R1CS-input openings, weighted by the `SpartanOuterPublic` coefficients.
pub struct OuterRemainder {
    shape: SpartanOuterDimensions,
}

impl SymbolicSumcheck for OuterRemainder {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = SpartanOuterDimensions;
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = crate::NoInputs<C>;
    type Outputs<C> = crate::NoOutputs<C>;

    fn new(shape: SpartanOuterDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::SpartanOuter
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.remainder_sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(outer_uniskip_opening())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let mut output = JoltExpr::zero();

        for (left_index, left_variable) in self.shape.variables().iter().copied().enumerate() {
            for (right_index, right_variable) in self.shape.variables().iter().copied().enumerate()
            {
                output = output
                    + derived(JoltDerivedId::from(
                        SpartanOuterPublic::QuadraticCoefficient {
                            left: left_index,
                            right: right_index,
                        },
                    )) * opening(outer_opening(left_variable))
                        * opening(outer_opening(right_variable));
            }
        }

        if self.shape.include_linear_terms() {
            for (index, variable) in self.shape.variables().iter().copied().enumerate() {
                output = output
                    + derived(JoltDerivedId::from(SpartanOuterPublic::LinearCoefficient(
                        index,
                    ))) * opening(outer_opening(variable));
            }
        }

        if self.shape.include_constant_term() {
            output = output + derived(JoltDerivedId::from(SpartanOuterPublic::ConstantCoefficient));
        }

        output
    }
}
