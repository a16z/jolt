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

use jolt_claims::protocols::jolt::{
    formulas::{dimensions::TraceDimensions, instruction},
    InstructionInputChallenge, InstructionInputPublic, JoltChallengeId, JoltPublicId,
    JoltRelationClaims, JoltRelationId,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::try_eq_mle;
use jolt_riscv::InstructionFlags;
use jolt_verifier_derive::{InputClaims, OutputClaims};
use serde::{Deserialize, Serialize};

use crate::stages::relations::{GetPoint, OpeningClaim, ConcreteSumcheck};
use crate::stages::stage2::Stage2ClearOutput;
use crate::VerifierError;

/// Produced instruction-input virtualization openings (the left/right operand
/// selector flags and their operand values), all sharing the single
/// instruction-input opening point. Generic over the cell.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(InstructionInputVirtualization)]
pub struct InstructionInputOutputClaims<C> {
    #[opening(InstructionFlags(InstructionFlags::LeftOperandIsRs1Value))]
    pub left_operand_is_rs1: C,
    #[opening(Rs1Value)]
    pub rs1_value: C,
    #[opening(InstructionFlags(InstructionFlags::LeftOperandIsPC))]
    pub left_operand_is_pc: C,
    #[opening(UnexpandedPC)]
    pub unexpanded_pc: C,
    #[opening(InstructionFlags(InstructionFlags::RightOperandIsRs2Value))]
    pub right_operand_is_rs2: C,
    #[opening(Rs2Value)]
    pub rs2_value: C,
    #[opening(InstructionFlags(InstructionFlags::RightOperandIsImm))]
    pub right_operand_is_imm: C,
    #[opening(Imm)]
    pub imm: C,
}

/// Consumed instruction-input openings: the left/right virtualized instruction
/// inputs reduced by stage 2's product remainder. The relation reads only these
/// values, so the input points are left empty. Generic over the cell.
#[derive(Clone, Debug, InputClaims)]
pub struct InstructionInputInputClaims<C> {
    #[opening(RightInstructionInput, from = SpartanProductVirtualization)]
    pub right_instruction_input: C,
    #[opening(LeftInstructionInput, from = SpartanProductVirtualization)]
    pub left_instruction_input: C,
}

impl<F: Field> InstructionInputInputClaims<OpeningClaim<F>> {
    /// Wire the consumed openings from stage 2's product-remainder left/right
    /// instruction inputs. Only the values feed the input claim (the output points
    /// come from this relation's own sumcheck point), so the input points are left
    /// empty.
    pub fn from_upstream(stage2: &Stage2ClearOutput<F>) -> Self {
        let value = |value: F| OpeningClaim {
            point: Vec::new(),
            value,
        };
        Self {
            right_instruction_input: value(
                stage2
                    .output_claims
                    .product_remainder
                    .right_instruction_input
                    .value,
            ),
            left_instruction_input: value(
                stage2
                    .output_claims
                    .product_remainder
                    .left_instruction_input
                    .value,
            ),
        }
    }
}

pub struct InstructionInput<F: Field> {
    symbolic: jolt_claims::protocols::jolt::relations::instruction::InputVirtualization,
    claims: JoltRelationClaims<F>,
    gamma: F,
    product_remainder_opening_point: Vec<F>,
}

impl<F: Field> InstructionInput<F> {
    pub fn new(
        trace_dimensions: TraceDimensions,
        gamma: F,
        product_remainder_opening_point: Vec<F>,
    ) -> Self {
        Self {
            claims: instruction::input_virtualization(trace_dimensions),
            symbolic: jolt_claims::protocols::jolt::relations::instruction::InputVirtualization::new(trace_dimensions),
            gamma,
            product_remainder_opening_point,
        }
    }
}

impl<F: Field> ConcreteSumcheck<F> for InstructionInput<F> {
    type Symbolic = jolt_claims::protocols::jolt::relations::instruction::InputVirtualization;
    type Inputs<C> = InstructionInputInputClaims<C>;
    type Outputs<C> = InstructionInputOutputClaims<C>;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn sumcheck_relation(&self) -> &JoltRelationClaims<F> {
        &self.claims
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &InstructionInputInputClaims<C>,
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

    fn resolve_challenge(&self, id: &JoltChallengeId) -> Result<F, VerifierError> {
        match id {
            JoltChallengeId::InstructionInput(InstructionInputChallenge::Gamma) => Ok(self.gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        }
    }

    fn resolve_public<C: GetPoint<F>>(
        &self,
        id: &JoltPublicId,
        _inputs: &InstructionInputInputClaims<C>,
        outputs: Option<&InstructionInputOutputClaims<OpeningClaim<F>>>,
    ) -> Result<F, VerifierError> {
        let outputs =
            outputs.ok_or(VerifierError::MissingStageClaimPublic { id: *id })?;
        let JoltPublicId::InstructionInput(public_id) = id else {
            return Err(VerifierError::MissingStageClaimPublic { id: *id });
        };
        match public_id {
            // Every instruction-input output shares the one opening point.
            InstructionInputPublic::EqProduct => try_eq_mle(
                outputs.unexpanded_pc.point(),
                &self.product_remainder_opening_point,
            )
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::InstructionInputVirtualization,
                reason: error.to_string(),
            }),
        }
    }
}
