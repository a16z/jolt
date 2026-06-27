//! Address phase of the two-phase advice claim-reduction symbolic relation.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use super::AdviceReductionShape;
use crate::protocols::jolt::geometry::claim_reductions::advice::{
    cycle_phase_advice_opening, final_advice_opening,
};
use crate::protocols::jolt::{
    AdviceClaimReductionPublic, JoltChallengeId, JoltExpr, JoltOpeningId,
    JoltDerivedId, JoltRelationId, JoltSumcheckSpec,
};
use crate::{opening, derived, InputClaims, OutputClaims, SymbolicSumcheck};

/// Produced final advice openings, keyed by kind; present only when that kind's
/// address phase ran. Generic over the cell.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(AdviceClaimReduction)]
pub struct AdviceAddressPhaseOutputClaims<C> {
    #[opening(trusted_advice)]
    pub trusted: Option<C>,
    #[opening(untrusted_advice)]
    pub untrusted: Option<C>,
}

/// Consumed cycle-phase advice openings, keyed by kind.
#[derive(Clone, Debug, InputClaims)]
pub struct AdviceAddressPhaseInputClaims<C> {
    #[opening(trusted_advice, from = AdviceClaimReductionCyclePhase)]
    pub trusted: Option<C>,
    #[opening(untrusted_advice, from = AdviceClaimReductionCyclePhase)]
    pub untrusted: Option<C>,
}

/// Address phase of the advice reduction: reduces the cycle-phase advice
/// opening to the final advice opening scaled by `FinalScale`.
pub struct AddressPhase {
    shape: AdviceReductionShape,
}

impl SymbolicSumcheck for AddressPhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = AdviceReductionShape;

    fn new(shape: AdviceReductionShape) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::AdviceClaimReduction
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.1.address_sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let kind = self.shape.0;
        opening(cycle_phase_advice_opening(kind))
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let kind = self.shape.0;
        derived(JoltDerivedId::from(AdviceClaimReductionPublic::FinalScale(
            kind,
        ))) * opening(final_advice_opening(kind))
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

    #[test]
    fn address_phase_evaluates_like_core_formula() {
        let relation = AddressPhase::new((JoltAdviceKind::Untrusted, with_address_phase()));

        let cycle_claim = Fr::from_u64(11);
        let final_advice_claim = Fr::from_u64(13);
        let final_scale = Fr::from_u64(17);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == cycle_phase_advice_opening(JoltAdviceKind::Untrusted) => cycle_claim,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );
        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == final_advice_opening(JoltAdviceKind::Untrusted) => final_advice_claim,
                _ => zero,
            },
            |_| zero,
            |id| match *id {
                JoltDerivedId::AdviceClaimReduction(AdviceClaimReductionPublic::FinalScale(
                    JoltAdviceKind::Untrusted,
                )) => final_scale,
                _ => zero,
            },
        );

        assert_eq!(input, cycle_claim);
        assert_eq!(output, final_scale * final_advice_claim);
    }

    #[test]
    fn address_phase_exposes_expected_dependencies() {
        let relation = AddressPhase::new((JoltAdviceKind::Trusted, with_address_phase()));

        assert_eq!(AddressPhase::id(), JoltRelationId::AdviceClaimReduction);
        assert_eq!(relation.spec(), with_address_phase().address_sumcheck());
        assert_eq!(
            relation.input_expression::<Fr>().required_openings(),
            vec![cycle_phase_advice_opening(JoltAdviceKind::Trusted)]
        );
        assert_eq!(
            relation.output_expression::<Fr>().required_openings(),
            vec![final_advice_opening(JoltAdviceKind::Trusted)]
        );
        assert_eq!(
            relation.required_deriveds::<Fr>(),
            vec![JoltDerivedId::from(AdviceClaimReductionPublic::FinalScale(
                JoltAdviceKind::Trusted
            ))]
        );
    }
}
