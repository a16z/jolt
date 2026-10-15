//! Cycle phase of the two-phase advice claim reduction, split per advice kind.
//!
//! The trusted and untrusted advice reductions are structurally identical but bind
//! disjoint openings, so each is its own relation type (`TrustedCyclePhase` /
//! `UntrustedCyclePhase`) with a single-slot claims pair — a non-`Option`
//! `trusted` / `untrusted` field. The static `#[opening]` id mapping per type makes
//! the off-kind slot unrepresentable: a trusted relation can only ever produce a
//! trusted-advice opening, so no runtime `kind → slot` match (with off-kind `None`
//! filling) is needed. `FinalScale` is keyed by the now type-fixed kind.

use jolt_field::Ring;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::advice::{
    cycle_phase_advice_opening, final_advice_opening, ram_val_check_advice_opening,
};
use crate::protocols::jolt::geometry::claim_reductions::precommitted::TWO_PHASE_DEGREE_BOUND;
use crate::protocols::jolt::{
    AdviceClaimReductionPublic, JoltAdviceKind, JoltChallengeId, JoltDerivedId, JoltExpr,
    JoltOpeningId, JoltRelationId, PrecommittedReductionDimensions,
};
use crate::{derived, opening, InputClaims, OutputClaims, SymbolicSumcheck};

/// The produced trusted-advice opening (the intermediate when an address phase
/// follows, else the final advice opening).
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(AdviceClaimReductionCyclePhase)]
pub struct TrustedAdviceCyclePhaseOutputClaims<C> {
    #[opening(trusted_advice)]
    pub trusted: C,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct TrustedAdviceCyclePhaseInputClaims<C> {
    #[opening(trusted_advice, from = RamValCheck)]
    pub trusted: C,
}

/// The produced untrusted-advice opening (the intermediate when an address phase
/// follows, else the final advice opening).
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(AdviceClaimReductionCyclePhase)]
pub struct UntrustedAdviceCyclePhaseOutputClaims<C> {
    #[opening(untrusted_advice)]
    pub untrusted: C,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct UntrustedAdviceCyclePhaseInputClaims<C> {
    #[opening(untrusted_advice, from = RamValCheck)]
    pub untrusted: C,
}

/// Cycle phase of the trusted-advice reduction: binds the RAM-val-check advice
/// opening to either the cycle-phase advice opening (when an address phase follows)
/// or directly to the final advice opening scaled by `FinalScale`.
#[derive(Clone)]
pub struct TrustedCyclePhase {
    dimensions: PrecommittedReductionDimensions,
}

impl SymbolicSumcheck for TrustedCyclePhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = PrecommittedReductionDimensions;
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = TrustedAdviceCyclePhaseInputClaims<C>;
    type Outputs<C> = TrustedAdviceCyclePhaseOutputClaims<C>;

    fn new(dimensions: PrecommittedReductionDimensions) -> Self {
        Self { dimensions }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::AdviceClaimReductionCyclePhase
    }

    fn rounds(&self) -> usize {
        self.dimensions.cycle_phase_total_rounds()
    }

    fn degree(&self) -> usize {
        TWO_PHASE_DEGREE_BOUND
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        opening(ram_val_check_advice_opening(JoltAdviceKind::Trusted))
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        if self.dimensions.has_address_phase() {
            opening(cycle_phase_advice_opening(JoltAdviceKind::Trusted))
        } else {
            derived(JoltDerivedId::from(AdviceClaimReductionPublic::FinalScale(
                JoltAdviceKind::Trusted,
            ))) * opening(final_advice_opening(JoltAdviceKind::Trusted))
        }
    }
}

/// Cycle phase of the untrusted-advice reduction: binds the RAM-val-check advice
/// opening to either the cycle-phase advice opening (when an address phase follows)
/// or directly to the final advice opening scaled by `FinalScale`.
#[derive(Clone)]
pub struct UntrustedCyclePhase {
    dimensions: PrecommittedReductionDimensions,
}

impl SymbolicSumcheck for UntrustedCyclePhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = PrecommittedReductionDimensions;
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = UntrustedAdviceCyclePhaseInputClaims<C>;
    type Outputs<C> = UntrustedAdviceCyclePhaseOutputClaims<C>;

    fn new(dimensions: PrecommittedReductionDimensions) -> Self {
        Self { dimensions }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::AdviceClaimReductionCyclePhase
    }

    fn rounds(&self) -> usize {
        self.dimensions.cycle_phase_total_rounds()
    }

    fn degree(&self) -> usize {
        TWO_PHASE_DEGREE_BOUND
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        opening(ram_val_check_advice_opening(JoltAdviceKind::Untrusted))
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        if self.dimensions.has_address_phase() {
            opening(cycle_phase_advice_opening(JoltAdviceKind::Untrusted))
        } else {
            derived(JoltDerivedId::from(AdviceClaimReductionPublic::FinalScale(
                JoltAdviceKind::Untrusted,
            ))) * opening(final_advice_opening(JoltAdviceKind::Untrusted))
        }
    }
}
