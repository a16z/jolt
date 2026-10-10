//! Address phase of the two-phase advice claim reduction, split per advice kind.
//!
//! The trusted and untrusted advice reductions are structurally identical but bind
//! disjoint openings, so each is its own relation type (`TrustedAddressPhase` /
//! `UntrustedAddressPhase`) with a single-slot claims pair — a non-`Option`
//! `trusted` / `untrusted` field. The static `#[opening]` id mapping per type makes
//! the off-kind slot unrepresentable: a trusted relation can only ever produce a
//! trusted-advice opening, so no runtime `kind → slot` match (with off-kind `None`
//! filling) is needed. `FinalScale` is keyed by the now type-fixed kind.

use jolt_field::Ring;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::advice::{
    cycle_phase_advice_opening, final_advice_opening,
};
use crate::protocols::jolt::geometry::claim_reductions::precommitted::TWO_PHASE_DEGREE_BOUND;
use crate::protocols::jolt::{
    AdviceClaimReductionPublic, JoltAdviceKind, JoltChallengeId, JoltDerivedId, JoltExpr,
    JoltOpeningId, JoltRelationId, PrecommittedReductionDimensions,
};
use crate::{derived, opening, InputClaims, OutputClaims, SymbolicSumcheck};

/// Produced final trusted-advice opening (scaled by `FinalScale`).
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(AdviceClaimReduction)]
pub struct TrustedAdviceAddressPhaseOutputClaims<C> {
    #[opening(trusted_advice)]
    pub trusted: C,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct TrustedAdviceAddressPhaseInputClaims<C> {
    #[opening(trusted_advice, from = AdviceClaimReductionCyclePhase)]
    pub trusted: C,
}

/// Produced final untrusted-advice opening (scaled by `FinalScale`).
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(AdviceClaimReduction)]
pub struct UntrustedAdviceAddressPhaseOutputClaims<C> {
    #[opening(untrusted_advice)]
    pub untrusted: C,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct UntrustedAdviceAddressPhaseInputClaims<C> {
    #[opening(untrusted_advice, from = AdviceClaimReductionCyclePhase)]
    pub untrusted: C,
}

/// Address phase of the trusted-advice reduction: reduces the cycle-phase advice
/// opening to the final advice opening scaled by `FinalScale`.
#[derive(Clone)]
pub struct TrustedAddressPhase {
    dimensions: PrecommittedReductionDimensions,
}

impl SymbolicSumcheck for TrustedAddressPhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = PrecommittedReductionDimensions;
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = TrustedAdviceAddressPhaseInputClaims<C>;
    type Outputs<C> = TrustedAdviceAddressPhaseOutputClaims<C>;

    fn new(dimensions: PrecommittedReductionDimensions) -> Self {
        Self { dimensions }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::AdviceClaimReduction
    }

    fn rounds(&self) -> usize {
        self.dimensions.address_phase_total_rounds()
    }

    fn degree(&self) -> usize {
        TWO_PHASE_DEGREE_BOUND
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        opening(cycle_phase_advice_opening(JoltAdviceKind::Trusted))
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        derived(JoltDerivedId::from(AdviceClaimReductionPublic::FinalScale(
            JoltAdviceKind::Trusted,
        ))) * opening(final_advice_opening(JoltAdviceKind::Trusted))
    }
}

/// Address phase of the untrusted-advice reduction: reduces the cycle-phase advice
/// opening to the final advice opening scaled by `FinalScale`.
#[derive(Clone)]
pub struct UntrustedAddressPhase {
    dimensions: PrecommittedReductionDimensions,
}

impl SymbolicSumcheck for UntrustedAddressPhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = PrecommittedReductionDimensions;
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = UntrustedAdviceAddressPhaseInputClaims<C>;
    type Outputs<C> = UntrustedAdviceAddressPhaseOutputClaims<C>;

    fn new(dimensions: PrecommittedReductionDimensions) -> Self {
        Self { dimensions }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::AdviceClaimReduction
    }

    fn rounds(&self) -> usize {
        self.dimensions.address_phase_total_rounds()
    }

    fn degree(&self) -> usize {
        TWO_PHASE_DEGREE_BOUND
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        opening(cycle_phase_advice_opening(JoltAdviceKind::Untrusted))
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        derived(JoltDerivedId::from(AdviceClaimReductionPublic::FinalScale(
            JoltAdviceKind::Untrusted,
        ))) * opening(final_advice_opening(JoltAdviceKind::Untrusted))
    }
}
