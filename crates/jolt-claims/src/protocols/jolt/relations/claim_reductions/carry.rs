//! Carry claim-reduction symbolic sumcheck relation (implicit-carry).
//!
//! Batches the committed `Carry` column's two sumcheck openings (product
//! virtualization and shift) with the `carry_init` all-zeros public pair
//! (claim 0, enforcing `Carry(0) = 0`) and reduces them to one final opening:
//!
//!   sum_t [eq(r_product, t) + gamma * eq(r_shift, t) + gamma^2 * eq(0, t)] * Carry(t)
//!     = carry_product + gamma * carry_shift + gamma^2 * 0

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::dimensions::TraceDimensions;
use crate::protocols::jolt::{
    CarryClaimReductionChallenge, CarryClaimReductionPublic, JoltChallengeId, JoltDerivedId,
    JoltExpr, JoltOpeningId, JoltRelationId,
};
use crate::{
    challenge, derived, opening, InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck,
};

/// The single reduced `Carry` opening at the reduction's own point.
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(CarryClaimReduction)]
pub struct CarryClaimReductionOutputClaims<C> {
    #[opening(committed = Carry)]
    pub carry: C,
}

/// The two consumed `Carry` openings. The `carry_init` pair is public
/// (all-zeros point, claim 0) and needs no wire cell.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct CarryClaimReductionInputClaims<C> {
    #[opening(committed = Carry, from = SpartanProductVirtualization)]
    pub carry_product: C,
    #[opening(committed = Carry, from = SpartanShift)]
    pub carry_shift: C,
}

/// Fiat-Shamir challenge drawn by the carry claim-reduction sumcheck.
#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct CarryClaimReductionChallenges<F> {
    #[challenge(CarryClaimReductionChallenge::Gamma)]
    pub gamma: F,
}

/// The carry claim-reduction sumcheck relation.
#[derive(Clone)]
pub struct CarryReduction {
    shape: TraceDimensions,
}

impl SymbolicSumcheck for CarryReduction {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = TraceDimensions;
    type Challenges<F> = CarryClaimReductionChallenges<F>;
    type Inputs<C> = CarryClaimReductionInputClaims<C>;
    type Outputs<C> = CarryClaimReductionOutputClaims<C>;

    fn new(shape: TraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::CarryClaimReduction
    }

    fn rounds(&self) -> usize {
        self.shape.log_t()
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(CarryClaimReductionChallenge::Gamma);
        opening(super::super::super::geometry::spartan::carry_product())
            + gamma * opening(super::super::super::geometry::spartan::carry_shift())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(CarryClaimReductionChallenge::Gamma);
        let combined_eq = derived(CarryClaimReductionPublic::EqCarryProduct)
            + gamma.clone() * derived(CarryClaimReductionPublic::EqCarryShift)
            + gamma.pow(2) * derived(CarryClaimReductionPublic::EqZeroSelector);
        combined_eq * opening(super::super::super::geometry::spartan::carry_reduced())
    }
}
