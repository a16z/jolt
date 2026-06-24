//! Two-phase advice claim-reduction symbolic relations (cycle -> address).

use jolt_field::RingCore;

use crate::protocols::jolt::formulas::claim_reductions::advice::{
    cycle_phase_advice_opening, final_advice_opening, ram_val_check_advice_opening,
};
use crate::protocols::jolt::{
    AdviceClaimReductionPublic, JoltAdviceKind, JoltChallengeId, JoltExpr, JoltOpeningId,
    JoltPublicId, JoltRelationId, JoltSumcheckSpec, PrecommittedReductionDimensions,
};
use crate::{opening, public, SymbolicSumcheck};

/// `(advice kind, two-phase dimensions)` shape shared by the advice cycle- and
/// address-phase reductions.
pub type AdviceReductionShape = (JoltAdviceKind, PrecommittedReductionDimensions);

/// Cycle phase of the advice reduction: binds the RAM-val-check advice opening
/// to either the cycle-phase advice opening (when an address phase follows) or
/// directly to the final advice opening scaled by `FinalScale`.
pub struct CyclePhase {
    shape: AdviceReductionShape,
}

impl SymbolicSumcheck for CyclePhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type PublicId = JoltPublicId;
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
            public(JoltPublicId::from(AdviceClaimReductionPublic::FinalScale(
                kind,
            ))) * opening(final_advice_opening(kind))
        }
    }
}

/// Address phase of the advice reduction: reduces the cycle-phase advice
/// opening to the final advice opening scaled by `FinalScale`.
pub struct AddressPhase {
    shape: AdviceReductionShape,
}

impl SymbolicSumcheck for AddressPhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type PublicId = JoltPublicId;
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
        public(JoltPublicId::from(AdviceClaimReductionPublic::FinalScale(
            kind,
        ))) * opening(final_advice_opening(kind))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
                JoltPublicId::AdviceClaimReduction(AdviceClaimReductionPublic::FinalScale(
                    JoltAdviceKind::Trusted,
                )) => final_scale,
                _ => zero,
            },
        );

        assert_eq!(input, input_advice);
        assert_eq!(output, final_scale * final_advice_claim);
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
                JoltPublicId::AdviceClaimReduction(AdviceClaimReductionPublic::FinalScale(
                    JoltAdviceKind::Untrusted,
                )) => final_scale,
                _ => zero,
            },
        );

        assert_eq!(input, cycle_claim);
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
        assert!(relation.required_publics::<Fr>().is_empty());
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
            relation.required_publics::<Fr>(),
            vec![JoltPublicId::from(AdviceClaimReductionPublic::FinalScale(
                JoltAdviceKind::Untrusted
            ))]
        );
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
            relation.required_publics::<Fr>(),
            vec![JoltPublicId::from(AdviceClaimReductionPublic::FinalScale(
                JoltAdviceKind::Trusted
            ))]
        );
    }
}
