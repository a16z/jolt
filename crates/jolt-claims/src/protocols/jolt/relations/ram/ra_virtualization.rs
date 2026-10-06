use jolt_field::Ring;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::ram::{
    committed_ram_ra_product, ram_ra_claim_reduction, RamRaVirtualizationDimensions,
};
use crate::protocols::jolt::{JoltExpr, JoltRelationId, RamRaVirtualizationPublic};
use crate::SymbolicSumcheck;
use crate::{derived, opening, InputClaims, OutputClaims};

#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RamRaVirtualization)]
pub struct RamRaVirtualizationOutputClaims<C> {
    #[opening(committed = RamRa)]
    pub ram_ra: Vec<C>,
}

/// The single reduced `RamRa` opening from the stage-5 RAM RA claim reduction.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct RamRaVirtualizationInputClaims<C> {
    #[opening(RamRa, from = RamRaClaimReduction)]
    pub ram_ra_reduced: C,
}

/// The RAM `ra` virtualization sumcheck: equates the reduced `ra` opening on the
/// input side with the product of the committed per-`d` `ra` openings, weighted
/// by the cycle-`eq` public, on the output side.
#[derive(Clone)]
pub struct RaVirtualization {
    shape: RamRaVirtualizationDimensions,
}

impl SymbolicSumcheck for RaVirtualization {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = RamRaVirtualizationDimensions;
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = RamRaVirtualizationInputClaims<C>;
    type Outputs<C> = RamRaVirtualizationOutputClaims<C>;

    fn new(shape: RamRaVirtualizationDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RamRaVirtualization
    }

    fn rounds(&self) -> usize {
        self.shape.log_t()
    }

    fn degree(&self) -> usize {
        self.shape.num_committed_ra_polys() + 1
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        opening(ram_ra_claim_reduction())
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        derived(RamRaVirtualizationPublic::EqCycle) * committed_ram_ra_product(self.shape)
    }
}
