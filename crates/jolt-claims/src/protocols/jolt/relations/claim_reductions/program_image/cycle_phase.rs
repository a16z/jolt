//! Cycle phase of the two-phase program-image (initial RAM) claim-reduction relation.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::program_image::{
    cycle_phase_program_image_opening, final_output_expr, ram_val_check_contribution_opening,
};
use crate::protocols::jolt::{
    JoltChallengeId, JoltExpr, JoltOpeningId, JoltDerivedId, JoltRelationId, JoltSumcheckSpec,
    PrecommittedReductionDimensions,
};
use crate::{opening, InputClaims, OutputClaims, SymbolicSumcheck};

/// The produced `ProgramImageInit` opening (intermediate or final).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(ProgramImageClaimReductionCyclePhase)]
pub struct ProgramImageReductionCyclePhaseOutputClaims<C> {
    #[opening(committed = ProgramImageInit)]
    pub program_image: C,
}

/// The consumed RAM value-check program-image contribution.
#[derive(Clone, Debug, InputClaims)]
pub struct ProgramImageReductionCyclePhaseInputClaims<C> {
    #[opening(ProgramImageInitContributionRw, from = RamValCheck)]
    pub contribution: C,
}

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
    type DerivedId = JoltDerivedId;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::claim_reductions::program_image::final_program_image_opening;
    use crate::protocols::jolt::ProgramImageClaimReductionPublic;
    use jolt_field::Fr;

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
        assert!(relation.required_deriveds::<Fr>().is_empty());
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
            relation.required_deriveds::<Fr>(),
            vec![JoltDerivedId::from(
                ProgramImageClaimReductionPublic::FinalScale
            )]
        );
    }
}
