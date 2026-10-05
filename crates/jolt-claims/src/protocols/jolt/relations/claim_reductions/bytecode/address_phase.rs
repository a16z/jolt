use jolt_field::Ring;
use serde::{Deserialize, Serialize};

use super::BytecodeReductionShape;
use crate::protocols::jolt::geometry::claim_reductions::bytecode::{
    assert_valid_chunk_count, cycle_phase_intermediate_opening, final_output_expr,
};
use crate::protocols::jolt::geometry::claim_reductions::precommitted::TWO_PHASE_DEGREE_BOUND;
use crate::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId,
};
use crate::{opening, InputClaims, OutputClaims, SymbolicSumcheck};

/// Produced per-chunk `BytecodeChunk(i)` openings, all sharing the reduction's
/// final opening point. Generic over the cell.
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(BytecodeClaimReduction)]
pub struct BytecodeReductionAddressPhaseOutputClaims<C> {
    #[opening(committed = BytecodeChunk)]
    pub chunks: Vec<C>,
}

/// Consumed intermediate opening from the stage-6b bytecode cycle phase.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct BytecodeReductionAddressPhaseInputClaims<C> {
    #[opening(BytecodeClaimReductionIntermediate, from = BytecodeClaimReductionCyclePhase)]
    pub cycle_phase_intermediate: C,
}

/// Address phase of the committed-bytecode reduction: reduces the cycle-phase
/// intermediate opening to the committed `BytecodeChunk(i)` openings weighted by
/// `ChunkOutputWeight`.
#[derive(Clone)]
pub struct AddressPhase {
    shape: BytecodeReductionShape,
}

impl SymbolicSumcheck for AddressPhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = BytecodeReductionShape;
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = BytecodeReductionAddressPhaseInputClaims<C>;
    type Outputs<C> = BytecodeReductionAddressPhaseOutputClaims<C>;

    fn new(shape: BytecodeReductionShape) -> Self {
        assert_valid_chunk_count(shape.1);
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::BytecodeClaimReduction
    }

    fn rounds(&self) -> usize {
        self.shape.0.address_phase_total_rounds()
    }

    fn degree(&self) -> usize {
        TWO_PHASE_DEGREE_BOUND
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        opening(cycle_phase_intermediate_opening())
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        final_output_expr(self.shape.1)
    }
}
