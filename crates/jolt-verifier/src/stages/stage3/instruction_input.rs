//! The stage 3 `InstructionInputVirtualization` sumcheck instance.
//!
//! Owns the instruction-input opening-point derivation and the `EqProduct`
//! public-value computation (against the product-remainder opening point).
//!
//! The reduced-vs-product input consistency guard — that stage 2's
//! `instruction_claim_reduction` left/right openings agree with the
//! product-remainder openings at the same point — is enforced by stage 2's
//! generated `validate_aliases` (driven by `InstructionClaimReduction`'s
//! declared aliases), which runs before any consumer wires these inputs.
//!
//! This relation's own `unexpanded_pc` output aliases the Spartan shift's: both
//! stage-3 members bind the same batch-point suffix (equal rounds, default
//! offsets) and derive the same reversed opening point. Declared in
//! `aliased_output_openings` below; the generated drivers absorb it via the
//! shift source and enforce the wire copy equals it.

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::instruction::{
    InstructionInputChallenges, InstructionInputInputClaims, InstructionInputOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::bytecode, geometry::dimensions::TraceDimensions, InstructionInputPublic,
    JoltDerivedId, JoltOpeningId, JoltRelationId,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::try_eq_mle;

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage2::Stage2BatchOutputClaims;
use crate::VerifierError;

/// Wire the consumed opening *values* from stage 2's product-remainder left/right
/// instruction inputs. Takes the ZK-agnostic stage-2 output-claims aggregate.
pub fn instruction_input_input_values_from_upstream<F: Field>(
    stage2: &Stage2BatchOutputClaims<F>,
) -> InstructionInputInputClaims<F> {
    let product_remainder = &stage2.product_remainder;
    InstructionInputInputClaims {
        right_instruction_input: product_remainder.right_instruction_input,
        left_instruction_input: product_remainder.left_instruction_input,
    }
}

#[derive(Clone)]
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

impl<F: Field> InstructionInput<F> {
    pub fn product_remainder_opening_point(&self) -> &[F] {
        &self.product_remainder_opening_point
    }
}

impl<F: Field> ConcreteSumcheck<F> for InstructionInput<F> {
    type Symbolic = relations::instruction::InputVirtualization;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn aliased_output_openings() -> Vec<(JoltOpeningId, JoltOpeningId)> {
        // The geometry pair is (shift, instruction-input); the shift opening is
        // the canonical source, so the aliased/source order swaps here.
        let [(shift_unexpanded_pc, instruction_unexpanded_pc)] =
            bytecode::read_raf_consistency_openings();
        vec![(instruction_unexpanded_pc, shift_unexpanded_pc)]
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
