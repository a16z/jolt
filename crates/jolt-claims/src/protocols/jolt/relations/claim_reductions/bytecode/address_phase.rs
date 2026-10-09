use jolt_field::Ring;
use serde::{Deserialize, Serialize};

use super::BytecodeReductionShape;
use crate::protocols::jolt::geometry::claim_reductions::bytecode::{
    cycle_phase_intermediate_opening, final_output_expr,
};
use crate::protocols::jolt::geometry::claim_reductions::precommitted::TWO_PHASE_DEGREE_BOUND;
use crate::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId,
};
use crate::{opening, InputClaims, OutputClaims, SymbolicSumcheck};

/// Final committed-bytecode claims at the reduction's opening point.
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(BytecodeClaimReduction)]
pub struct BytecodeReductionAddressPhaseOutputClaims<C> {
    #[cfg(not(feature = "akita"))]
    #[opening(committed = BytecodeChunk)]
    pub chunks: Vec<C>,
    #[cfg(feature = "akita")]
    #[opening(committed = ProgramBytecode)]
    pub bytecode: C,
}

/// Consumed intermediate opening from the stage-6b bytecode cycle phase.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct BytecodeReductionAddressPhaseInputClaims<C> {
    #[opening(BytecodeClaimReductionIntermediate, from = BytecodeClaimReductionCyclePhase)]
    pub cycle_phase_intermediate: C,
}

/// Address phase of the committed-bytecode reduction: reduces the cycle-phase
/// intermediate opening to weighted committed-bytecode claims.
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
        #[cfg(not(feature = "akita"))]
        crate::protocols::jolt::geometry::claim_reductions::bytecode::assert_valid_chunk_count(
            shape.1,
        );
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::BytecodeClaimReduction
    }

    fn rounds(&self) -> usize {
        {
            #[cfg(not(feature = "akita"))]
            let dimensions = self.shape.0;
            #[cfg(feature = "akita")]
            let dimensions = self.shape;
            dimensions.address_phase_total_rounds()
        }
    }

    fn degree(&self) -> usize {
        TWO_PHASE_DEGREE_BOUND
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        opening(cycle_phase_intermediate_opening())
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        final_output_expr(
            #[cfg(not(feature = "akita"))]
            self.shape.1,
        )
    }
}
