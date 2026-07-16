//! Reduces the fused-increment claim from the standalone virtualization point
//! to the cycle point shared by the stage-6b batch.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::{
    FusedIncClaimReductionPublic, JoltExpr, JoltOpeningId, JoltRelationId, JoltVirtualPolynomial,
    TraceDimensions,
};
use crate::{derived, opening, InputClaims, OutputClaims, SymbolicSumcheck};

use super::inc_virtualization::fused_inc_opening;

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct FusedIncClaimReductionInputClaims<C> {
    #[opening(FusedInc, from = IncVirtualization)]
    pub fused_inc: C,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(FusedIncClaimReduction)]
pub struct FusedIncClaimReductionOutputClaims<C> {
    #[opening(FusedInc)]
    pub fused_inc: C,
}

pub struct FusedIncClaimReduction {
    shape: TraceDimensions,
}

impl SymbolicSumcheck for FusedIncClaimReduction {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = TraceDimensions;
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = FusedIncClaimReductionInputClaims<C>;
    type Outputs<C> = FusedIncClaimReductionOutputClaims<C>;

    fn new(shape: TraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::FusedIncClaimReduction
    }

    fn rounds(&self) -> usize {
        self.shape.log_t()
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(fused_inc_opening())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        derived(FusedIncClaimReductionPublic::EqIncVirtualization)
            * opening(fused_inc_reduced_opening())
    }
}

pub fn fused_inc_reduced_opening() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::FusedInc,
        JoltRelationId::FusedIncClaimReduction,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::JoltDerivedId;
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn relation_reduces_the_virtualized_fused_increment() {
        let relation = FusedIncClaimReduction::new(TraceDimensions::new(5));
        let input = Fr::from_u64(3);
        let output = Fr::from_u64(5);
        let eq = Fr::from_u64(7);
        let zero = Fr::from_u64(0);

        assert_eq!(
            relation.input_expression::<Fr>().evaluate(
                |id| {
                    if *id == fused_inc_opening() {
                        input
                    } else {
                        zero
                    }
                },
                |_| zero,
                |_| zero,
            ),
            input
        );
        assert_eq!(
            relation.output_expression::<Fr>().evaluate(
                |id| {
                    if *id == fused_inc_reduced_opening() {
                        output
                    } else {
                        zero
                    }
                },
                |_| zero,
                |id| match *id {
                    JoltDerivedId::FusedIncClaimReduction(
                        FusedIncClaimReductionPublic::EqIncVirtualization,
                    ) => eq,
                    _ => zero,
                },
            ),
            eq * output
        );
    }
}
