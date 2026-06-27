//! Address phase of the two-phase program-image (initial RAM) claim-reduction relation.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::program_image::{
    cycle_phase_program_image_opening, final_output_expr,
};
use crate::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId, JoltSumcheckSpec,
    PrecommittedReductionDimensions,
};
use crate::{opening, InputClaims, OutputClaims, SymbolicSumcheck};

/// Produced `ProgramImageInit` opening at the reduction's final opening point.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(ProgramImageClaimReduction)]
pub struct ProgramImageReductionAddressPhaseOutputClaims<C> {
    #[opening(committed = ProgramImageInit)]
    pub program_image: C,
}

/// Consumed intermediate opening from the stage-6b program-image cycle phase.
#[derive(Clone, Debug, InputClaims)]
pub struct ProgramImageReductionAddressPhaseInputClaims<C> {
    #[opening(committed = ProgramImageInit, from = ProgramImageClaimReductionCyclePhase)]
    pub cycle_phase: C,
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
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = PrecommittedReductionDimensions;
    type Challenges<F> = crate::NoChallenges<F>;

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
                JoltDerivedId::ProgramImageClaimReduction(
                    ProgramImageClaimReductionPublic::FinalScale,
                ) => final_scale,
                _ => zero,
            },
        );

        assert_eq!(input, intermediate);
        assert_eq!(output, final_scale * final_claim);
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
            relation.required_deriveds::<Fr>(),
            vec![JoltDerivedId::from(
                ProgramImageClaimReductionPublic::FinalScale
            )]
        );
    }
}
