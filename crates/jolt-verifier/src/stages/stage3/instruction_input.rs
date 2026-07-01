//! The stage 3 `InstructionInputVirtualization` sumcheck instance.
//!
//! A self-contained relation object driven identically by the prover (while
//! producing the stage 3 batch proof) and the verifier (after checking it). It
//! owns the instruction-input opening-point derivation and the `EqProduct`
//! public-value computation (against the product-remainder opening point), so the
//! input/output claim algebra lives here once.
//!
//! The reduced-vs-product input consistency guard — that stage 2's
//! `instruction_claim_reduction` left/right openings (when present) agree with the
//! product-remainder openings at the same point — lives in
//! [`Stage2BatchOutputClaims::validate`](crate::stages::stage2::Stage2BatchOutputClaims::validate),
//! which the stage-2 verifier runs before any consumer wires these inputs.

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::instruction::{
    InstructionInputChallenges, InstructionInputInputClaims, InstructionInputOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::dimensions::TraceDimensions, InstructionInputPublic, JoltDerivedId, JoltRelationId,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::try_eq_mle;

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage2::Stage2ClearOutput;
use crate::VerifierError;

/// Wire the consumed opening *values* from stage 2's product-remainder left/right
/// instruction inputs. (Verifier-side constructor for the moved
/// [`InstructionInputInputClaims`].)
pub fn instruction_input_input_values_from_upstream<F: Field>(
    stage2: &Stage2ClearOutput<F>,
) -> InstructionInputInputClaims<F> {
    let product_remainder = &stage2.output_values.product_remainder;
    InstructionInputInputClaims {
        right_instruction_input: product_remainder.right_instruction_input,
        left_instruction_input: product_remainder.left_instruction_input,
    }
}

/// Wire the consumed opening *points* from stage 2. Only the values feed the input
/// claim (the output points come from this relation's own sumcheck point), so the
/// input points are left empty.
pub fn instruction_input_input_points_from_upstream<F: Field>(
    _stage2: &Stage2ClearOutput<F>,
) -> InstructionInputInputClaims<Vec<F>> {
    InstructionInputInputClaims {
        right_instruction_input: Vec::new(),
        left_instruction_input: Vec::new(),
    }
}

pub struct InstructionInput<F: Field> {
    symbolic: relations::instruction::InputVirtualization,
    product_remainder_opening_point: Vec<F>,
}

impl<F: Field> InstructionInput<F> {
    pub fn new(trace_dimensions: TraceDimensions, product_remainder_opening_point: Vec<F>) -> Self {
        Self {
            symbolic: relations::instruction::InputVirtualization::new(trace_dimensions),
            product_remainder_opening_point,
        }
    }
}

impl<F: Field> ConcreteSumcheck<F> for InstructionInput<F> {
    type Symbolic = relations::instruction::InputVirtualization;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &InstructionInputInputClaims<Vec<F>>,
    ) -> Result<InstructionInputOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        Ok(InstructionInputOutputClaims {
            left_operand_is_rs1: opening_point.clone(),
            rs1_value: opening_point.clone(),
            left_operand_is_pc: opening_point.clone(),
            unexpanded_pc: opening_point.clone(),
            right_operand_is_rs2: opening_point.clone(),
            rs2_value: opening_point.clone(),
            right_operand_is_imm: opening_point.clone(),
            imm: opening_point,
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &InstructionInputInputClaims<Vec<F>>,
        output_points: &InstructionInputOutputClaims<Vec<F>>,
        _challenges: &InstructionInputChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::InstructionInput(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        match public_id {
            // Every instruction-input output shares the one opening point.
            InstructionInputPublic::EqProduct => try_eq_mle(
                output_points.unexpanded_pc(),
                &self.product_remainder_opening_point,
            )
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::InstructionInputVirtualization,
                reason: error.to_string(),
            }),
        }
    }
}
