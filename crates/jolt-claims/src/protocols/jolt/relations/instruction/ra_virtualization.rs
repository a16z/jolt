use jolt_field::Ring;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::instruction::{
    committed_instruction_ra_product, weighted_instruction_ra_sum,
    InstructionRaVirtualizationDimensions,
};
use crate::protocols::jolt::{
    InstructionRaVirtualizationChallenge, InstructionRaVirtualizationPublic, JoltExpr,
    JoltRelationId,
};
use crate::SymbolicSumcheck;
use crate::{challenge, derived, InputClaims, OutputClaims, SumcheckChallenges};

#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(InstructionRaVirtualization)]
pub struct InstructionRaVirtualizationOutputClaims<C> {
    #[opening(committed = InstructionRa)]
    pub committed_instruction_ra: Vec<C>,
}

/// The per-virtual reduced `InstructionRa` openings from the stage-5 instruction
/// read-RAF.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct InstructionRaVirtualizationInputClaims<C> {
    #[opening(InstructionRa, from = InstructionReadRaf)]
    pub instruction_ra: Vec<C>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct InstructionRaVirtualizationChallenges<F> {
    #[challenge(InstructionRaVirtualizationChallenge::Gamma)]
    pub gamma: F,
}

/// The instruction RA-virtualization sumcheck: relates the virtual
/// instruction-RA openings (folded by `gamma`) to the per-virtual products of
/// committed instruction-RA openings, weighted by the `EqCycle` public.
#[derive(Clone)]
pub struct RaVirtualization {
    shape: InstructionRaVirtualizationDimensions,
}

impl SymbolicSumcheck for RaVirtualization {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = InstructionRaVirtualizationDimensions;
    type Challenges<F> = InstructionRaVirtualizationChallenges<F>;
    type Inputs<C> = InstructionRaVirtualizationInputClaims<C>;
    type Outputs<C> = InstructionRaVirtualizationOutputClaims<C>;

    fn new(shape: InstructionRaVirtualizationDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::InstructionRaVirtualization
    }

    fn rounds(&self) -> usize {
        self.shape.log_t()
    }

    fn degree(&self) -> usize {
        self.shape.num_committed_per_virtual() + 1
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        let gamma = challenge(InstructionRaVirtualizationChallenge::Gamma);
        weighted_instruction_ra_sum(self.shape, gamma)
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        let gamma = challenge(InstructionRaVirtualizationChallenge::Gamma);
        let eq_cycle = derived(InstructionRaVirtualizationPublic::EqCycle);
        let mut output = JoltExpr::zero();
        for virtual_index in 0..self.shape.num_virtual_ra_polys() {
            output = output
                + eq_cycle.clone()
                    * gamma.clone().pow(virtual_index)
                    * committed_instruction_ra_product(self.shape, virtual_index);
        }
        output
    }
}
