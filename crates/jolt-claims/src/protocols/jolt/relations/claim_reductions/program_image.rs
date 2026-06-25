//! Two-phase program-image (initial RAM) claim-reduction symbolic relations.

use jolt_field::RingCore;

use crate::protocols::jolt::geometry::claim_reductions::program_image::{
    cycle_phase_program_image_opening, final_output_expr, ram_val_check_contribution_opening,
};
use crate::protocols::jolt::{
    JoltChallengeId, JoltExpr, JoltOpeningId, JoltPublicId, JoltRelationId, JoltSumcheckSpec,
    PrecommittedReductionDimensions,
};
use crate::{opening, SymbolicSumcheck};

/// Cycle phase of the program-image reduction: binds the staged
/// `ProgramImageInitContributionRw` scalar to either the cycle-phase
/// intermediate opening (when an address phase follows) or directly to the
/// final committed `ProgramImageInit` opening scaled by `FinalScale`.
pub struct CyclePhase {
    shape: PrecommittedReductionDimensions,
}

impl SymbolicSumcheck for CyclePhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type PublicId = JoltPublicId;
    type ChallengeId = JoltChallengeId;
    type Shape = PrecommittedReductionDimensions;

    fn new(shape: PrecommittedReductionDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::ProgramImageClaimReductionCyclePhase
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.cycle_sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(ram_val_check_contribution_opening())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        if self.shape.has_address_phase() {
            opening(cycle_phase_program_image_opening())
        } else {
            final_output_expr()
        }
    }
}

/// Address phase of the program-image reduction: reduces the cycle-phase
/// intermediate opening to the final committed `ProgramImageInit` opening
/// scaled by `FinalScale`.
pub struct AddressPhase {
    shape: PrecommittedReductionDimensions,
}

impl SymbolicSumcheck for AddressPhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type PublicId = JoltPublicId;
    type ChallengeId = JoltChallengeId;
    type Shape = PrecommittedReductionDimensions;

    fn new(shape: PrecommittedReductionDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::ProgramImageClaimReduction
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.address_sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(cycle_phase_program_image_opening())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        final_output_expr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::claim_reductions::program_image::final_program_image_opening;
    use crate::protocols::jolt::ProgramImageClaimReductionPublic;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    #[test]
    fn address_phase_evaluates_like_core_formula() {
        let dimensions = PrecommittedReductionDimensions::new(4, 3, true);
        let relation = AddressPhase::new(dimensions);

        let intermediate = fr(11);
        let final_claim = fr(13);
        let final_scale = fr(17);
        let zero = fr(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| {
                if *id == cycle_phase_program_image_opening() {
                    intermediate
                } else {
                    zero
                }
            },
            |_| zero,
            |_| zero,
        );
        let output = relation.output_expression::<Fr>().evaluate(
            |id| {
                if *id == final_program_image_opening() {
                    final_claim
                } else {
                    zero
                }
            },
            |_| zero,
            |id| match *id {
                JoltPublicId::ProgramImageClaimReduction(
                    ProgramImageClaimReductionPublic::FinalScale,
                ) => final_scale,
                _ => zero,
            },
        );

        assert_eq!(input, intermediate);
        assert_eq!(output, final_scale * final_claim);
    }

    #[test]
    fn cycle_phase_with_address_phase_exposes_expected_dependencies() {
        let dimensions = PrecommittedReductionDimensions::new(4, 3, true);
        let relation = CyclePhase::new(dimensions);

        assert_eq!(
            CyclePhase::id(),
            JoltRelationId::ProgramImageClaimReductionCyclePhase
        );
        assert_eq!(relation.spec(), dimensions.cycle_sumcheck());
        assert_eq!(
            relation.input_expression::<Fr>().required_openings(),
            vec![ram_val_check_contribution_opening()]
        );
        assert_eq!(
            relation.output_expression::<Fr>().required_openings(),
            vec![cycle_phase_program_image_opening()]
        );
        assert!(relation.required_challenges::<Fr>().is_empty());
        assert!(relation.required_publics::<Fr>().is_empty());
    }

    #[test]
    fn cycle_phase_without_address_phase_exposes_final_scale() {
        let dimensions = PrecommittedReductionDimensions::new(4, 3, false);
        let relation = CyclePhase::new(dimensions);

        assert_eq!(
            relation.output_expression::<Fr>().required_openings(),
            vec![final_program_image_opening()]
        );
        assert_eq!(
            relation.required_publics::<Fr>(),
            vec![JoltPublicId::from(
                ProgramImageClaimReductionPublic::FinalScale
            )]
        );
    }

    #[test]
    fn address_phase_exposes_expected_dependencies() {
        let dimensions = PrecommittedReductionDimensions::new(4, 3, true);
        let relation = AddressPhase::new(dimensions);

        assert_eq!(
            AddressPhase::id(),
            JoltRelationId::ProgramImageClaimReduction
        );
        assert_eq!(relation.spec(), dimensions.address_sumcheck());
        assert_eq!(
            relation.input_expression::<Fr>().required_openings(),
            vec![cycle_phase_program_image_opening()]
        );
        assert_eq!(
            relation.output_expression::<Fr>().required_openings(),
            vec![final_program_image_opening()]
        );
        assert_eq!(
            relation.required_publics::<Fr>(),
            vec![JoltPublicId::from(
                ProgramImageClaimReductionPublic::FinalScale
            )]
        );
    }
}
