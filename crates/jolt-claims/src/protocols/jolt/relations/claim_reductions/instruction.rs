use jolt_field::Ring;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::instruction::{
    left_instruction_input_reduced, left_instruction_input_spartan, left_lookup_operand_reduced,
    left_lookup_operand_spartan, lookup_output_reduced, lookup_output_spartan,
    right_instruction_input_reduced, right_instruction_input_spartan, right_lookup_operand_reduced,
    right_lookup_operand_spartan, weighted_claims,
};
use crate::protocols::jolt::{
    InstructionClaimReductionChallenge, InstructionClaimReductionPublic, JoltChallengeId,
    JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId, TraceDimensions,
};
use crate::{derived, InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck};

/// Produced reduced instruction-lookup openings, all sharing the single reduced
/// opening point. Generic over the cell. Field declaration order is the canonical
/// Fiat-Shamir order (single-sourced via [`OutputClaims::canonical_order`]).
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(InstructionClaimReduction)]
pub struct InstructionClaimReductionOutputClaims<C> {
    #[opening(LookupOutput)]
    pub lookup_output: C,
    #[opening(LeftLookupOperand)]
    pub left_lookup_operand: C,
    #[opening(RightLookupOperand)]
    pub right_lookup_operand: C,
    #[opening(LeftInstructionInput)]
    pub left_instruction_input: C,
    #[opening(RightInstructionInput)]
    pub right_instruction_input: C,
}

/// Consumed instruction-lookup openings from stage 1's outer sumcheck, reduced by
/// this sumcheck. The relation reads only these values (its output point comes from
/// its own sumcheck point), so the input points are left empty. Generic over the
/// cell. Field order matches the generated stage-2 batch declaration.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct InstructionClaimReductionInputClaims<C> {
    #[opening(LookupOutput, from = SpartanOuter)]
    pub lookup_output: C,
    #[opening(LeftLookupOperand, from = SpartanOuter)]
    pub left_lookup_operand: C,
    #[opening(RightLookupOperand, from = SpartanOuter)]
    pub right_lookup_operand: C,
    #[opening(LeftInstructionInput, from = SpartanOuter)]
    pub left_instruction_input: C,
    #[opening(RightInstructionInput, from = SpartanOuter)]
    pub right_instruction_input: C,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct InstructionClaimReductionChallenges<F> {
    #[challenge(InstructionClaimReductionChallenge::Gamma)]
    pub gamma: F,
}

/// Batches the Spartan-outer instruction-lookup openings (lookup output, left/
/// right lookup operands, left/right instruction inputs) by `gamma` and reduces
/// them to the instruction-claim-reduction openings weighted by `EqSpartan`.
#[derive(Clone)]
pub struct ClaimReduction {
    shape: TraceDimensions,
}

impl SymbolicSumcheck for ClaimReduction {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = TraceDimensions;
    type Challenges<F> = InstructionClaimReductionChallenges<F>;
    type Inputs<C> = InstructionClaimReductionInputClaims<C>;
    type Outputs<C> = InstructionClaimReductionOutputClaims<C>;

    fn new(shape: TraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::InstructionClaimReduction
    }

    fn rounds(&self) -> usize {
        self.shape.log_t()
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        weighted_claims(
            lookup_output_spartan(),
            left_lookup_operand_spartan(),
            right_lookup_operand_spartan(),
            left_instruction_input_spartan(),
            right_instruction_input_spartan(),
        )
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        derived(InstructionClaimReductionPublic::EqSpartan)
            * weighted_claims(
                lookup_output_reduced(),
                left_lookup_operand_reduced(),
                right_lookup_operand_reduced(),
                left_instruction_input_reduced(),
                right_instruction_input_reduced(),
            )
    }
}
