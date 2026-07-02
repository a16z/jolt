//! Unsigned increment claim reduction: shifts the fused increment into the
//! unsigned range and re-binds its cycle point to the shared one-hot cycle
//! point.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::{
    JoltExpr, JoltOpeningId, JoltRelationId, JoltVirtualPolynomial, TraceDimensions,
};
use crate::{constant, opening, InputClaims, NoChallenges, OutputClaims, SymbolicSumcheck};

use super::super::geometry::UNSIGNED_INC_BITS;
use super::inc_virtualization::fused_inc_opening;

/// The unsigned fused increment at the shared cycle point. Its low
/// [`UNSIGNED_INC_BITS`] bits are decoded by the chunk columns and its top bit
/// by the msb column (see `UnsignedIncChunkReconstruction`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(UnsignedIncClaimReduction)]
pub struct UnsignedIncReductionOutputClaims<C> {
    #[opening(UnsignedInc)]
    pub unsigned_inc: C,
}

#[derive(Clone, Debug, InputClaims)]
pub struct UnsignedIncReductionInputClaims<C> {
    #[opening(FusedInc, from = IncVirtualization)]
    pub fused_inc: C,
}

/// `UnsignedInc = FusedInc + 2^64`: the signed increment (in `(-2^64, 2^64)`)
/// shifted into `(0, 2^65)` so a base-`2^b` one-hot decomposition can carry
/// it. The sumcheck re-binds the cycle point from the virtualization's bound
/// point to the booleanity/hamming cycle point that all packed one-hot columns
/// share.
pub struct UnsignedIncReduction {
    shape: TraceDimensions,
}

impl SymbolicSumcheck for UnsignedIncReduction {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = TraceDimensions;
    type Challenges<F> = NoChallenges<F>;
    type Inputs<C> = UnsignedIncReductionInputClaims<C>;
    type Outputs<C> = UnsignedIncReductionOutputClaims<C>;

    fn new(shape: TraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::UnsignedIncClaimReduction
    }

    fn rounds(&self) -> usize {
        self.shape.log_t()
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(fused_inc_opening()) + constant(F::pow2(UNSIGNED_INC_BITS))
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(unsigned_inc_opening())
    }
}

pub fn unsigned_inc_opening() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::UnsignedInc,
        JoltRelationId::UnsignedIncClaimReduction,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt, RingCore};

    #[test]
    fn reduction_shifts_by_two_to_the_sixty_four() {
        let relation = UnsignedIncReduction::new(TraceDimensions::new(5));
        let fused_inc = Fr::from_u64(41);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == fused_inc_opening() => fused_inc,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );
        assert_eq!(input, fused_inc + Fr::pow2(64));

        // A negative increment lands in (0, 2^64): the msb of the unsigned
        // value is zero and the low chunks carry `2^64 - |inc|`.
        let negative = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == fused_inc_opening() => zero - Fr::from_u64(41),
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );
        assert_eq!(negative, Fr::from_u128((1u128 << 64) - 41));
    }

    #[test]
    fn reduction_exposes_expected_dependencies() {
        let relation = UnsignedIncReduction::new(TraceDimensions::new(5));

        assert_eq!(
            UnsignedIncReduction::id(),
            JoltRelationId::UnsignedIncClaimReduction
        );
        assert_eq!(relation.rounds(), 5);
        assert_eq!(relation.degree(), 2);
        assert_eq!(
            relation.input_expression::<Fr>().required_openings(),
            vec![fused_inc_opening()]
        );
        assert_eq!(
            relation.output_expression::<Fr>().required_openings(),
            vec![unsigned_inc_opening()]
        );
        assert!(relation.required_challenges::<Fr>().is_empty());
        assert!(relation.required_deriveds::<Fr>().is_empty());
    }
}
