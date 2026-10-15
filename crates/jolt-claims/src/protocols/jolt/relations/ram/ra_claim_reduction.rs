use jolt_field::Ring;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::ram::{
    ram_ra, ram_ra_claim_reduction, ram_ra_raf_evaluation, ram_ra_val_check,
};
use crate::protocols::jolt::{
    JoltExpr, JoltRelationId, RamRaClaimReductionChallenge, RamRaClaimReductionPublic,
    TraceDimensions,
};
use crate::SymbolicSumcheck;
use crate::{challenge, derived, opening, InputClaims, OutputClaims, SumcheckChallenges};

/// Produced RAM-RA reduced opening, generic over the opening cell (`F` for the
/// serialized wire value, `Vec<F>` for the derived opening point).
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RamRaClaimReduction)]
pub struct RamRaClaimReductionOutputClaims<C> {
    #[opening(RamRa)]
    pub ram_ra: C,
}

/// Consumed RAM-RA openings reduced by the `RamRaClaimReduction` sumcheck, wired
/// from the upstream RAF-evaluation, read-write-checking, and val-check
/// relations. Generic over the opening cell (`F` for the serialized wire value,
/// `Vec<F>` for the derived opening point).
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct RamRaClaimReductionInputClaims<C> {
    #[opening(RamRa, from = RamRafEvaluation)]
    pub raf: C,
    #[opening(RamRa, from = RamReadWriteChecking)]
    pub read_write: C,
    #[opening(RamRa, from = RamValCheck)]
    pub val_check: C,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct RamRaClaimReductionChallenges<F> {
    #[challenge(RamRaClaimReductionChallenge::Gamma)]
    pub gamma: F,
}

/// The RAM `ra` claim-reduction sumcheck: folds the three `ra` openings (RAF,
/// read/write, val-check) by `gamma` on the input side, and matches the reduced
/// `ra` opening weighted by the matching cycle-`eq` publics on the output side.
#[derive(Clone)]
pub struct RaClaimReduction {
    shape: TraceDimensions,
}

impl SymbolicSumcheck for RaClaimReduction {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = TraceDimensions;
    type Challenges<F> = RamRaClaimReductionChallenges<F>;
    type Inputs<C> = RamRaClaimReductionInputClaims<C>;
    type Outputs<C> = RamRaClaimReductionOutputClaims<C>;

    fn new(shape: TraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RamRaClaimReduction
    }

    fn rounds(&self) -> usize {
        self.shape.log_t()
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        let gamma = challenge(RamRaClaimReductionChallenge::Gamma);
        opening(ram_ra_raf_evaluation())
            + gamma.clone() * opening(ram_ra())
            + gamma.clone().pow(2) * opening(ram_ra_val_check())
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        let gamma = challenge(RamRaClaimReductionChallenge::Gamma);
        (derived(RamRaClaimReductionPublic::EqCycleRaf)
            + gamma.clone() * derived(RamRaClaimReductionPublic::EqCycleReadWrite)
            + gamma.pow(2) * derived(RamRaClaimReductionPublic::EqCycleValCheck))
            * opening(ram_ra_claim_reduction())
    }
}
