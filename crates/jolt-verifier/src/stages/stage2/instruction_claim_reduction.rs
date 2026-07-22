//! The stage 2 `InstructionClaimReduction` sumcheck instance.
//!
//! Owns the reduced-claim opening-point derivation and the `EqSpartan` public-value
//! computation, in lockstep with the BlindFold constraint's
//! `claim_reductions::instruction::claim_reduction` formula.
//!
//! Three of the five reduced openings (`lookup_output`, `left_instruction_input`,
//! `right_instruction_input`) alias the corresponding
//! `SpartanProductVirtualization` product-remainder openings — the reduction and
//! the product remainder bind the same batch-point suffix (equal rounds, default
//! offsets) and derive the same reversed opening point, so each pair is the same
//! polynomial at the same point. The aliases are declared once, in
//! [`aliased_output_openings`](crate::stages::relations::ConcreteSumcheck::aliased_output_openings)
//! below; the generated drivers absorb each aliased opening via its
//! product-remainder source and enforce the wire copies equal it.

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::claim_reductions::instruction::{
    InstructionClaimReductionChallenges, InstructionClaimReductionInputClaims,
    InstructionClaimReductionOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::dimensions::TraceDimensions, geometry::instruction, InstructionClaimReductionPublic,
    JoltDerivedId, JoltOpeningId, JoltRelationId,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::try_eq_mle;

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage1::Stage1ClearOutput;
use crate::VerifierError;

/// Wire the consumed instruction-lookup opening *values* from stage 1's outer
/// sumcheck. (Verifier-side constructor for the moved
/// [`InstructionClaimReductionInputClaims`] — it reads the verifier-only
/// [`Stage1ClearOutput`], so it cannot live in `jolt-claims`.)
pub fn instruction_claim_reduction_input_values_from_upstream<F: Field>(
    stage1: &Stage1ClearOutput<F>,
) -> InstructionClaimReductionInputClaims<F> {
    let outer = &stage1.output_values.outer_remainder;
    InstructionClaimReductionInputClaims {
        lookup_output: outer.lookup_output,
        left_lookup_operand: outer.left_lookup_operand,
        right_lookup_operand: outer.right_lookup_operand,
        left_instruction_input: outer.left_instruction_input,
        right_instruction_input: outer.right_instruction_input,
    }
}

#[derive(Clone)]
pub struct InstructionClaimReduction<F: Field> {
    symbolic: relations::claim_reductions::instruction::ClaimReduction,
    tau_low: Vec<F>,
}

impl<F: Field> InstructionClaimReduction<F> {
    pub fn new(trace_dimensions: TraceDimensions, tau_low: Vec<F>) -> Self {
        Self {
            symbolic: relations::claim_reductions::instruction::ClaimReduction::new(
                trace_dimensions,
            ),
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

impl<F: Field> InstructionClaimReduction<F> {
    pub fn tau_low(&self) -> &[F] {
        &self.tau_low
    }
}

impl<F: Field> ConcreteSumcheck<F> for InstructionClaimReduction<F> {
    type Symbolic = relations::claim_reductions::instruction::ClaimReduction;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn aliased_output_openings() -> Vec<(JoltOpeningId, JoltOpeningId)> {
        let [lookup_output] = instruction::read_raf_consistency_openings();
        let [left_input, right_input] = instruction::input_virtualization_consistency_openings();
        vec![lookup_output, left_input, right_input]
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &InstructionClaimReductionInputClaims<Vec<F>>,
    ) -> Result<InstructionClaimReductionOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        Ok(InstructionClaimReductionOutputClaims {
            lookup_output: opening_point.clone(),
            left_lookup_operand: opening_point.clone(),
            right_lookup_operand: opening_point.clone(),
            left_instruction_input: opening_point.clone(),
            right_instruction_input: opening_point,
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &InstructionClaimReductionInputClaims<Vec<F>>,
        output_points: &InstructionClaimReductionOutputClaims<Vec<F>>,
        _challenges: &InstructionClaimReductionChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::InstructionClaimReduction(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        match public_id {
            // The reduced openings share one opening point; bind it against the low
            // product remainder challenges (`tau_low`).
            InstructionClaimReductionPublic::EqSpartan => {
                try_eq_mle(output_points.left_lookup_operand(), &self.tau_low)
                    .map_err(public_input_failed)
            }
        }
    }
}
