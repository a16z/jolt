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

    fn sumcheck(&self) -> JoltSumcheckSpec {
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

    fn sumcheck(&self) -> JoltSumcheckSpec {
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
    use jolt_field::Fr;

    fn with_address_phase() -> PrecommittedReductionDimensions {
        PrecommittedReductionDimensions::new(4, 3, true)
    }

    fn without_address_phase() -> PrecommittedReductionDimensions {
        PrecommittedReductionDimensions::new(4, 3, false)
    }

    #[test]
    fn cycle_phase_with_address_phase_exposes_expected_dependencies() {
        let relation = CyclePhase::new((JoltAdviceKind::Trusted, with_address_phase()));

        assert_eq!(
            CyclePhase::id(),
            JoltRelationId::AdviceClaimReductionCyclePhase
        );
        assert_eq!(relation.sumcheck(), with_address_phase().cycle_sumcheck());
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
        assert_eq!(relation.sumcheck(), with_address_phase().address_sumcheck());
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
