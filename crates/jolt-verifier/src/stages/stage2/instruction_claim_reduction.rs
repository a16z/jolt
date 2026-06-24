//! The stage 2 `InstructionClaimReduction` sumcheck instance.
//!
//! A self-contained relation object driven identically by the prover (while
//! producing the stage 2 batch proof) and the verifier (after checking it). It
//! owns the reduced-claim opening-point derivation and the `EqSpartan` public-value
//! computation, so the input/output claim algebra lives here once (and stays in
//! lockstep with the BlindFold constraint, which evaluates the same
//! `claim_reductions::instruction::claim_reduction` formula).
//!
//! WARNING — cross-relation aliases: three of the five reduced openings
//! (`lookup_output`, `left_instruction_input`, `right_instruction_input`) are not
//! re-committed when the reduction shares the product-remainder opening point; they
//! alias the corresponding `SpartanProductVirtualization` product-remainder
//! openings. They are therefore [`Option`] on the wire (absent ⇒ aliased), and the
//! opening-claims helper fills them from the product-remainder openings (or zero
//! when the points disagree) before this relation's output `Expr` is evaluated.

use jolt_claims::protocols::jolt::{
    formulas::{
        claim_reductions::instruction as instruction_claim_reduction, dimensions::TraceDimensions,
    },
    InstructionClaimReductionChallenge, InstructionClaimReductionPublic, JoltChallengeId,
    JoltPublicId, JoltRelationClaims, JoltRelationId,
};
use jolt_field::Field;
use jolt_poly::try_eq_mle;
use jolt_verifier_derive::{InputClaims, OutputClaims};
use serde::{Deserialize, Serialize};

use crate::stages::relations::{GetPoint, OpeningClaim, ConcreteSumcheck};
use crate::stages::stage1::Stage1ClearOutput;
use crate::VerifierError;

/// Produced reduced instruction-lookup openings, all sharing the single reduced
/// opening point. The three aliased openings are [`Option`] (absent on the wire ⇒
/// they alias the product-remainder openings; the opening-claims helper fills
/// them). Generic over the cell. Field order is the canonical Fiat-Shamir order
/// and must match [`instruction_claim_reduction::claim_reduction_output_openings`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(InstructionClaimReduction)]
pub struct InstructionClaimReductionOutputClaims<C> {
    #[opening(LookupOutput)]
    pub lookup_output: Option<C>,
    #[opening(LeftLookupOperand)]
    pub left_lookup_operand: C,
    #[opening(RightLookupOperand)]
    pub right_lookup_operand: C,
    #[opening(LeftInstructionInput)]
    pub left_instruction_input: Option<C>,
    #[opening(RightInstructionInput)]
    pub right_instruction_input: Option<C>,
}

/// Consumed instruction-lookup openings from stage 1's outer sumcheck, reduced by
/// this sumcheck. The relation reads only these values (its output point comes from
/// its own sumcheck point), so the input points are left empty. Generic over the
/// cell. Field order matches
/// [`instruction_claim_reduction::claim_reduction_input_openings`].
#[derive(Clone, Debug, InputClaims)]
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

impl<F: Field> InstructionClaimReductionInputClaims<OpeningClaim<F>> {
    pub fn from_upstream(stage1: &Stage1ClearOutput<F>) -> Self {
        let value = |value: F| OpeningClaim {
            point: Vec::new(),
            value,
        };
        Self {
            lookup_output: value(stage1.outer.lookup_output),
            left_lookup_operand: value(stage1.outer.left_lookup_operand),
            right_lookup_operand: value(stage1.outer.right_lookup_operand),
            left_instruction_input: value(stage1.outer.left_instruction_input),
            right_instruction_input: value(stage1.outer.right_instruction_input),
        }
    }
}

pub struct InstructionClaimReduction<F: Field> {
    claims: JoltRelationClaims<F>,
    gamma: F,
    tau_low: Vec<F>,
}

impl<F: Field> InstructionClaimReduction<F> {
    pub fn new(trace_dimensions: TraceDimensions, gamma: F, tau_low: Vec<F>) -> Self {
        Self {
            claims: instruction_claim_reduction::claim_reduction(trace_dimensions),
            gamma,
            tau_low,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::InstructionClaimReduction,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for InstructionClaimReduction<F> {
    type Inputs<C> = InstructionClaimReductionInputClaims<C>;
    type Outputs<C> = InstructionClaimReductionOutputClaims<C>;

    fn sumcheck_relation(&self) -> &JoltRelationClaims<F> {
        &self.claims
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &InstructionClaimReductionInputClaims<C>,
    ) -> Result<InstructionClaimReductionOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        Ok(InstructionClaimReductionOutputClaims {
            lookup_output: Some(opening_point.clone()),
            left_lookup_operand: opening_point.clone(),
            right_lookup_operand: opening_point.clone(),
            left_instruction_input: Some(opening_point.clone()),
            right_instruction_input: Some(opening_point),
        })
    }

    fn resolve_challenge(&self, id: &JoltChallengeId) -> Result<F, VerifierError> {
        match id {
            JoltChallengeId::InstructionClaimReduction(
                InstructionClaimReductionChallenge::Gamma,
            ) => Ok(self.gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        }
    }

    fn resolve_public<C: GetPoint<F>>(
        &self,
        id: &JoltPublicId,
        _inputs: &InstructionClaimReductionInputClaims<C>,
        outputs: Option<&InstructionClaimReductionOutputClaims<OpeningClaim<F>>>,
    ) -> Result<F, VerifierError> {
        let outputs =
            outputs.ok_or(VerifierError::MissingStageClaimPublic { id: *id })?;
        let JoltPublicId::InstructionClaimReduction(public_id) = id else {
            return Err(VerifierError::MissingStageClaimPublic { id: *id });
        };
        match public_id {
            // The reduced openings share one opening point; bind it against the low
            // product remainder challenges (`tau_low`).
            InstructionClaimReductionPublic::EqSpartan => {
                try_eq_mle(outputs.left_lookup_operand.point(), &self.tau_low)
                    .map_err(public_input_failed)
            }
        }
    }
}
