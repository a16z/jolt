use jolt_field::Ring;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::precommitted::TWO_PHASE_DEGREE_BOUND;
use crate::protocols::jolt::geometry::claim_reductions::program_image::{
    cycle_phase_program_image_opening, final_output_expr,
};
use crate::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId,
    PrecommittedReductionDimensions,
};
use crate::{opening, InputClaims, OutputClaims, SymbolicSumcheck};

/// Produced `ProgramImageInit` opening at the reduction's final opening point.
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
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
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct ProgramImageReductionAddressPhaseInputClaims<C> {
    #[opening(committed = ProgramImageInit, from = ProgramImageClaimReductionCyclePhase)]
    pub cycle_phase: C,
}

/// Address phase of the program-image reduction: reduces the cycle-phase
/// intermediate opening to the final committed `ProgramImageInit` opening
/// scaled by `FinalScale`.
#[derive(Clone)]
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
    type Inputs<C> = ProgramImageReductionAddressPhaseInputClaims<C>;
    type Outputs<C> = ProgramImageReductionAddressPhaseOutputClaims<C>;

    fn new(shape: PrecommittedReductionDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::ProgramImageClaimReduction
    }

    fn rounds(&self) -> usize {
        self.shape.address_phase_total_rounds()
    }

    fn degree(&self) -> usize {
        TWO_PHASE_DEGREE_BOUND
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        opening(cycle_phase_program_image_opening())
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        final_output_expr()
    }
}
