use jolt_field::Ring;
use serde::{Deserialize, Serialize};

use super::BytecodeReductionShape;
use crate::protocols::jolt::geometry::claim_reductions::bytecode::{
    assert_valid_chunk_count, bytecode_val_stage_opening, cycle_phase_intermediate_opening,
    final_output_expr, NUM_BYTECODE_VAL_STAGES,
};
use crate::protocols::jolt::geometry::claim_reductions::precommitted::TWO_PHASE_DEGREE_BOUND;
use crate::protocols::jolt::{
    BytecodeClaimReductionChallenge, JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId,
    JoltRelationId,
};
use crate::{challenge, opening, InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck};

/// The produced bytecode-reduction openings: the intermediate when an address
/// phase follows, else the per-chunk final `BytecodeChunk` openings.
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(BytecodeClaimReductionCyclePhase)]
pub struct BytecodeReductionCyclePhaseOutputClaims<C> {
    #[opening(BytecodeClaimReductionIntermediate)]
    pub intermediate: Option<C>,
    #[opening(committed = BytecodeChunk)]
    pub chunks: Vec<C>,
}

/// The consumed staged `BytecodeValClaim` openings from the bytecode read-RAF
/// address phase.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct BytecodeReductionCyclePhaseInputClaims<C> {
    #[opening(BytecodeValClaim, from = BytecodeReadRaf)]
    pub val_stages: Vec<C>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct BytecodeReductionCyclePhaseChallenges<F> {
    #[challenge(BytecodeClaimReductionChallenge::Eta)]
    pub eta: F,
}

/// Cycle phase of the committed-bytecode reduction: batches the staged
/// `BytecodeValClaim(i)` openings by powers of `eta` and reduces them to either
/// the cycle-phase intermediate opening (when an address phase follows) or the
/// committed `BytecodeChunk(i)` openings weighted by `ChunkOutputWeight`.
#[derive(Clone)]
pub struct CyclePhase {
    shape: BytecodeReductionShape,
}

impl SymbolicSumcheck for CyclePhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = BytecodeReductionShape;
    type Challenges<F> = BytecodeReductionCyclePhaseChallenges<F>;
    type Inputs<C> = BytecodeReductionCyclePhaseInputClaims<C>;
    type Outputs<C> = BytecodeReductionCyclePhaseOutputClaims<C>;

    fn new(shape: BytecodeReductionShape) -> Self {
        assert_valid_chunk_count(shape.1);
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::BytecodeClaimReductionCyclePhase
    }

    fn rounds(&self) -> usize {
        self.shape.0.cycle_phase_total_rounds()
    }

    fn degree(&self) -> usize {
        TWO_PHASE_DEGREE_BOUND
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        let eta = challenge(BytecodeClaimReductionChallenge::Eta);
        let mut input = JoltExpr::zero();
        for stage in 0..NUM_BYTECODE_VAL_STAGES {
            input = input + eta.clone().pow(stage) * opening(bytecode_val_stage_opening(stage));
        }
        input
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        let (dimensions, chunk_count) = self.shape;
        if dimensions.has_address_phase() {
            opening(cycle_phase_intermediate_opening())
        } else {
            final_output_expr(chunk_count)
        }
    }
}
