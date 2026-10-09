#[cfg(feature = "akita")]
use crate::MissingOpeningValue;
#[cfg(feature = "akita")]
use jolt_field::JoltField;
use jolt_field::Ring;
use serde::{Deserialize, Serialize};

#[cfg(feature = "akita")]
use super::BytecodeReductionAddressPhaseOutputClaims;
use super::BytecodeReductionShape;
use crate::protocols::jolt::geometry::claim_reductions::bytecode::{
    bytecode_val_stage_opening, cycle_phase_intermediate_opening, final_output_expr,
    NUM_BYTECODE_VAL_STAGES,
};
use crate::protocols::jolt::geometry::claim_reductions::precommitted::TWO_PHASE_DEGREE_BOUND;
use crate::protocols::jolt::{
    BytecodeClaimReductionChallenge, JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId,
    JoltRelationId,
};
use crate::{challenge, opening, InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck};

/// The produced bytecode-reduction openings: the intermediate when an address
/// phase follows, else the per-chunk final `BytecodeChunk` openings.
#[cfg(not(feature = "akita"))]
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

#[cfg(feature = "akita")]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
pub enum BytecodeReductionCyclePhaseOutputClaims<C> {
    Intermediate(BytecodeReductionIntermediateClaims<C>),
    Final(BytecodeReductionAddressPhaseOutputClaims<C>),
}

#[cfg(feature = "akita")]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(BytecodeClaimReductionCyclePhase)]
pub struct BytecodeReductionIntermediateClaims<C> {
    #[opening(BytecodeClaimReductionIntermediate)]
    pub intermediate: C,
}

#[cfg(feature = "akita")]
impl<C> BytecodeReductionCyclePhaseOutputClaims<C> {
    pub fn intermediate(&self) -> Option<&C> {
        match self {
            Self::Intermediate(claim) => Some(&claim.intermediate),
            Self::Final(_) => None,
        }
    }
    pub fn bytecode(&self) -> Option<&C> {
        match self {
            Self::Final(claim) => Some(&claim.bytecode),
            Self::Intermediate(_) => None,
        }
    }
}

// OutputClaims derives support structs; the enum delegates each exclusive state
// to its derived carrier so opening identities and order still have one owner.
#[cfg(feature = "akita")]
impl<F: JoltField> OutputClaims<F> for BytecodeReductionCyclePhaseOutputClaims<F> {
    fn canonical_order(&self) -> Vec<JoltOpeningId> {
        match self {
            Self::Intermediate(c) => c.canonical_order(),
            Self::Final(c) => c.canonical_order(),
        }
    }
    fn resolve_output(&self, id: &JoltOpeningId) -> Option<F> {
        match self {
            Self::Intermediate(c) => c.resolve_output(id),
            Self::Final(c) => c.resolve_output(id),
        }
    }
    fn from_opening_values(
        mut resolve: impl FnMut(&JoltOpeningId) -> Option<F>,
    ) -> Result<Self, MissingOpeningValue<JoltOpeningId>> {
        if let Some(intermediate) = resolve(&cycle_phase_intermediate_opening()) {
            Ok(Self::Intermediate(BytecodeReductionIntermediateClaims {
                intermediate,
            }))
        } else {
            BytecodeReductionAddressPhaseOutputClaims::from_opening_values(resolve).map(Self::Final)
        }
    }
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
        #[cfg(not(feature = "akita"))]
        crate::protocols::jolt::geometry::claim_reductions::bytecode::assert_valid_chunk_count(
            shape.1,
        );
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::BytecodeClaimReductionCyclePhase
    }

    fn rounds(&self) -> usize {
        {
            #[cfg(not(feature = "akita"))]
            let dimensions = self.shape.0;
            #[cfg(feature = "akita")]
            let dimensions = self.shape;
            dimensions.cycle_phase_total_rounds()
        }
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
        #[cfg(not(feature = "akita"))]
        let (dimensions, chunk_count) = self.shape;
        #[cfg(feature = "akita")]
        let dimensions = self.shape;
        if dimensions.has_address_phase() {
            opening(cycle_phase_intermediate_opening())
        } else {
            final_output_expr(
                #[cfg(not(feature = "akita"))]
                chunk_count,
            )
        }
    }
}
