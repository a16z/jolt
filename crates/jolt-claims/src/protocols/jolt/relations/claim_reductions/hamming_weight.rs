use jolt_field::Ring;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::hamming_weight::{
    booleanity_claim, hamming_weight_claim, reduced_claim, virtualization_claim,
    HammingWeightClaimReductionDimensions,
};
use crate::protocols::jolt::{
    HammingWeightClaimReductionChallenge, HammingWeightClaimReductionPublic, JoltChallengeId,
    JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId,
};
use crate::{
    challenge, derived, opening, InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck,
};

/// Produced one-hot `Ra` opening claims, grouped by family (instruction,
/// bytecode, RAM) in canonical layout order. Every produced opening shares the
/// single hamming-weight opening point. Generic over the opening cell (`F` for
/// the serialized wire value, `Vec<F>` for the derived opening point).
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(HammingWeightClaimReduction)]
pub struct HammingWeightClaimReductionOutputClaims<C> {
    #[opening(committed = InstructionRa)]
    pub instruction_ra: Vec<C>,
    #[opening(committed = BytecodeRa)]
    pub bytecode_ra: Vec<C>,
    #[opening(committed = RamRa)]
    pub ram_ra: Vec<C>,
}

/// Consumed claims reduced by the hamming-weight sumcheck: the RAM hamming-weight
/// claim (from RAM hamming booleanity) plus the per-family booleanity and
/// virtualization claims (each wired from its producing stage-6 relation).
/// Generic over the cell.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct HammingWeightClaimReductionInputClaims<C> {
    #[opening(RamHammingWeight, from = RamHammingBooleanity)]
    pub ram_hamming_weight: C,
    #[opening(committed = InstructionRa, from = Booleanity)]
    pub instruction_booleanity: Vec<C>,
    #[opening(committed = BytecodeRa, from = Booleanity)]
    pub bytecode_booleanity: Vec<C>,
    #[opening(committed = RamRa, from = Booleanity)]
    pub ram_booleanity: Vec<C>,
    #[opening(committed = InstructionRa, from = InstructionRaVirtualization)]
    pub instruction_virtualization: Vec<C>,
    #[opening(committed = BytecodeRa, from = BytecodeReadRaf)]
    pub bytecode_virtualization: Vec<C>,
    #[opening(committed = RamRa, from = RamRaVirtualization)]
    pub ram_virtualization: Vec<C>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct HammingWeightClaimReductionChallenges<F> {
    #[challenge(HammingWeightClaimReductionChallenge::Gamma)]
    pub gamma: F,
}

/// Batches each RA polynomial's hamming-weight, booleanity, and virtualization
/// claims by powers of `gamma` and reduces them to the per-polynomial
/// hamming-weight-claim-reduction openings weighted by the eq publics.
#[derive(Clone)]
pub struct ClaimReduction {
    shape: HammingWeightClaimReductionDimensions,
}

impl SymbolicSumcheck for ClaimReduction {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = HammingWeightClaimReductionDimensions;
    type Challenges<F> = HammingWeightClaimReductionChallenges<F>;
    type Inputs<C> = HammingWeightClaimReductionInputClaims<C>;
    type Outputs<C> = HammingWeightClaimReductionOutputClaims<C>;

    fn new(shape: HammingWeightClaimReductionDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::HammingWeightClaimReduction
    }

    fn rounds(&self) -> usize {
        self.shape.log_k_chunk
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        let gamma = challenge(HammingWeightClaimReductionChallenge::Gamma);
        let mut input = JoltExpr::zero();

        for (i, polynomial) in self.shape.layout.polynomials().enumerate() {
            input = input
                + gamma.clone().pow(3 * i) * hamming_weight_claim(polynomial)
                + gamma.clone().pow(3 * i + 1) * opening(booleanity_claim(polynomial))
                + gamma.clone().pow(3 * i + 2) * opening(virtualization_claim(polynomial));
        }

        input
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        let gamma = challenge(HammingWeightClaimReductionChallenge::Gamma);
        let mut output = JoltExpr::zero();

        for (i, polynomial) in self.shape.layout.polynomials().enumerate() {
            let output_coeff = gamma.clone().pow(3 * i)
                + gamma.clone().pow(3 * i + 1)
                    * derived(HammingWeightClaimReductionPublic::EqBooleanity)
                + gamma.clone().pow(3 * i + 2)
                    * derived(HammingWeightClaimReductionPublic::EqVirtualization(i));
            output = output + output_coeff * opening(reduced_claim(polynomial));
        }

        output
    }
}
