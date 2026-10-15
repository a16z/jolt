use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::registers::{
    rd_write_value_reduced, rd_write_value_spartan, rs1_value_reduced, rs1_value_spartan,
    rs2_value_reduced, rs2_value_spartan,
};
use crate::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltOpeningId, JoltRelationId,
    RegistersClaimReductionChallenge, RegistersClaimReductionPublic, TraceDimensions,
};
use crate::twist::claim_reductions as twist;
use crate::{InputClaims, OutputClaims, SumcheckChallenges};

/// Produced register claim-reduction openings (`rd` write value, `rs1`/`rs2`
/// values reduced to the Spartan point), all sharing the single reduction opening
/// point. Generic over the cell.
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RegistersClaimReduction)]
pub struct RegistersClaimReductionOutputClaims<C> {
    #[opening(RdWriteValue)]
    pub rd_write_value: C,
    #[opening(Rs1Value)]
    pub rs1_value: C,
    #[opening(Rs2Value)]
    pub rs2_value: C,
}

/// Consumed register openings reduced by this sumcheck, wired from stage 1's outer
/// sumcheck. The relation reads only these values, so the input points are left
/// empty. Generic over the cell.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct RegistersClaimReductionInputClaims<C> {
    #[opening(RdWriteValue, from = SpartanOuter)]
    pub rd_write_value: C,
    #[opening(Rs1Value, from = SpartanOuter)]
    pub rs1_value: C,
    #[opening(Rs2Value, from = SpartanOuter)]
    pub rs2_value: C,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct RegistersClaimReductionChallenges<F> {
    #[challenge(RegistersClaimReductionChallenge::Gamma)]
    pub gamma: F,
}

/// Batches the Spartan-outer register openings (`RdWriteValue`, `Rs1Value`,
/// `Rs2Value`) by `gamma` and reduces them to the registers-claim-reduction
/// openings weighted by the `EqSpartan` public.
#[derive(Clone)]
pub struct ClaimReduction {
    shape: TraceDimensions,
}

twist::instantiate_value_reduction! {
    relation = ClaimReduction,
    id = JoltRelationId::RegistersClaimReduction,
    ids = (JoltRelationId, JoltOpeningId, JoltDerivedId, JoltChallengeId),
    dimensions = TraceDimensions,
    challenges = RegistersClaimReductionChallenges,
    inputs = RegistersClaimReductionInputClaims,
    outputs = RegistersClaimReductionOutputClaims,
    consumed = [
        rd_write_value_spartan(),
        rs1_value_spartan(),
        rs2_value_spartan(),
    ],
    reduced = [
        rd_write_value_reduced(),
        rs1_value_reduced(),
        rs2_value_reduced(),
    ],
    gamma = RegistersClaimReductionChallenge::Gamma,
    eq_spartan = RegistersClaimReductionPublic::EqSpartan,
}
