use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::increments::{
    ram_inc_reduced, rd_inc_reduced,
};
use crate::protocols::jolt::geometry::ram::{ram_inc, ram_inc_val_check};
use crate::protocols::jolt::geometry::registers::{rd_inc_read_write, rd_inc_val_evaluation};
use crate::protocols::jolt::{
    IncClaimReductionChallenge, IncClaimReductionPublic, JoltChallengeId, JoltDerivedId,
    JoltOpeningId, JoltRelationId, TraceDimensions,
};
use crate::twist::claim_reductions as twist;
use crate::{InputClaims, OutputClaims, SumcheckChallenges};

#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(IncClaimReduction)]
pub struct IncClaimReductionOutputClaims<C> {
    #[opening(committed = RamInc)]
    pub ram_inc: C,
    #[opening(committed = RdInc)]
    pub rd_inc: C,
}

/// The four reduced `Inc` openings consumed from the read-write / value
/// relations of RAM and registers.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct IncClaimReductionInputClaims<C> {
    #[opening(committed = RamInc, from = RamReadWriteChecking)]
    pub ram_inc_read_write: C,
    #[opening(committed = RamInc, from = RamValCheck)]
    pub ram_inc_val_check: C,
    #[opening(committed = RdInc, from = RegistersReadWriteChecking)]
    pub rd_inc_read_write: C,
    #[opening(committed = RdInc, from = RegistersValEvaluation)]
    pub rd_inc_val_evaluation: C,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct IncClaimReductionChallenges<F> {
    #[challenge(IncClaimReductionChallenge::Gamma)]
    pub gamma: F,
}

/// Batches the RAM/register increment openings (`RamInc` read-write and
/// val-check, `RdInc` read-write and val-evaluation) by `gamma` and reduces
/// them to the increment-claim-reduction openings weighted by the eq publics.
#[derive(Clone)]
pub struct ClaimReduction {
    shape: TraceDimensions,
}

// RAM group first, registers group second — the relation's γ offset order.
// The lattice `read_raf` fold consumes the same four openings in this order at
// its own gamma powers.
twist::instantiate_increment_reduction! {
    relation = ClaimReduction,
    id = JoltRelationId::IncClaimReduction,
    ids = (JoltRelationId, JoltOpeningId, JoltDerivedId, JoltChallengeId),
    dimensions = TraceDimensions,
    challenges = IncClaimReductionChallenges,
    inputs = IncClaimReductionInputClaims,
    outputs = IncClaimReductionOutputClaims,
    groups = vec![
        twist::IncrementReductionGroup {
            consumed: [ram_inc(), ram_inc_val_check()],
            eq_publics: [
                IncClaimReductionPublic::EqRamReadWrite.into(),
                IncClaimReductionPublic::EqRamValCheck.into(),
            ],
            reduced: ram_inc_reduced(),
        },
        twist::IncrementReductionGroup {
            consumed: [rd_inc_read_write(), rd_inc_val_evaluation()],
            eq_publics: [
                IncClaimReductionPublic::EqRegistersReadWrite.into(),
                IncClaimReductionPublic::EqRegistersValEvaluation.into(),
            ],
            reduced: rd_inc_reduced(),
        },
    ],
    gamma = IncClaimReductionChallenge::Gamma,
}
