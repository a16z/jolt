//! Cycle phase of the two-phase advice claim-reduction symbolic relation.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use super::AdviceReductionShape;
use crate::protocols::jolt::geometry::claim_reductions::advice::{
    cycle_phase_advice_opening, final_advice_opening, ram_val_check_advice_opening,
};
use crate::protocols::jolt::{
    AdviceClaimReductionPublic, JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId,
    JoltRelationId, JoltSumcheckSpec,
};
use crate::{derived, opening, InputClaims, OutputClaims, SymbolicSumcheck};

/// The produced advice opening (the intermediate when an address phase follows,
/// else the final advice opening), keyed by kind.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(AdviceClaimReductionCyclePhase)]
pub struct AdviceCyclePhaseOutputClaims<C> {
    #[opening(trusted_advice)]
    pub trusted: Option<C>,
    #[opening(untrusted_advice)]
    pub untrusted: Option<C>,
}

/// The consumed RAM value-check advice opening, keyed by kind.
#[derive(Clone, Debug, InputClaims)]
pub struct AdviceCyclePhaseInputClaims<C> {
    #[opening(trusted_advice, from = RamValCheck)]
    pub trusted: Option<C>,
    #[opening(untrusted_advice, from = RamValCheck)]
    pub untrusted: Option<C>,
}

/// Cycle phase of the advice reduction: binds the RAM-val-check advice opening
/// to either the cycle-phase advice opening (when an address phase follows) or
/// directly to the final advice opening scaled by `FinalScale`.
pub struct CyclePhase {
    shape: AdviceReductionShape,
}

impl SymbolicSumcheck for CyclePhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = AdviceReductionShape;

    fn new(shape: AdviceReductionShape) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::AdviceClaimReductionCyclePhase
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.1.cycle_sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let kind = self.shape.0;
        opening(ram_val_check_advice_opening(kind))
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let (kind, dimensions) = self.shape;
        if dimensions.has_address_phase() {
            opening(cycle_phase_advice_opening(kind))
        } else {
            derived(JoltDerivedId::from(AdviceClaimReductionPublic::FinalScale(
                kind,
            ))) * opening(final_advice_opening(kind))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::{JoltAdviceKind, PrecommittedReductionDimensions};
    use jolt_field::{Fr, FromPrimitiveInt};

    fn with_address_phase() -> PrecommittedReductionDimensions {
        PrecommittedReductionDimensions::new(4, 3, true)
    }

    fn without_address_phase() -> PrecommittedReductionDimensions {
        PrecommittedReductionDimensions::new(4, 3, false)
    }

    #[test]
    fn cycle_phase_without_address_phase_evaluates_like_core_formula() {
        let relation = CyclePhase::new((JoltAdviceKind::Trusted, without_address_phase()));

        let input_advice = Fr::from_u64(3);
        let final_advice_claim = Fr::from_u64(5);
        let final_scale = Fr::from_u64(7);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == ram_val_check_advice_opening(JoltAdviceKind::Trusted) => input_advice,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );
        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == final_advice_opening(JoltAdviceKind::Trusted) => final_advice_claim,
                _ => zero,
            },
            |_| zero,
            |id| match *id {
                JoltDerivedId::AdviceClaimReduction(AdviceClaimReductionPublic::FinalScale(
                    JoltAdviceKind::Trusted,
                )) => final_scale,
                _ => zero,
            },
        );

        assert_eq!(input, input_advice);
        assert_eq!(output, final_scale * final_advice_claim);
    }

    #[test]
    fn cycle_phase_with_address_phase_exposes_expected_dependencies() {
        let relation = CyclePhase::new((JoltAdviceKind::Trusted, with_address_phase()));

        assert_eq!(
            CyclePhase::id(),
            JoltRelationId::AdviceClaimReductionCyclePhase
        );
        assert_eq!(relation.spec(), with_address_phase().cycle_sumcheck());
        assert_eq!(
            relation.input_expression::<Fr>().required_openings(),
            vec![ram_val_check_advice_opening(JoltAdviceKind::Trusted)]
        );
        assert_eq!(
            relation.output_expression::<Fr>().required_openings(),
            vec![cycle_phase_advice_opening(JoltAdviceKind::Trusted)]
        );
        assert!(relation.required_challenges::<Fr>().is_empty());
        assert!(relation.required_deriveds::<Fr>().is_empty());
    }

    #[test]
    fn cycle_phase_without_address_phase_exposes_final_scale() {
        let relation = CyclePhase::new((JoltAdviceKind::Untrusted, without_address_phase()));

        assert_eq!(
            relation.required_openings::<Fr>(),
            vec![
                ram_val_check_advice_opening(JoltAdviceKind::Untrusted),
                final_advice_opening(JoltAdviceKind::Untrusted),
            ]
        );
        assert_eq!(
            relation.required_deriveds::<Fr>(),
            vec![JoltDerivedId::from(AdviceClaimReductionPublic::FinalScale(
                JoltAdviceKind::Untrusted
            ))]
        );
    }
}
