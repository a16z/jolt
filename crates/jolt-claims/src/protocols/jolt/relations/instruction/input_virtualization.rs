use jolt_riscv::InstructionFlags;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::instruction::INPUT_VIRTUALIZATION_DEGREE;
use crate::protocols::jolt::{
    InstructionInputChallenge, InstructionInputPublic, JoltExpr, JoltRelationId,
    JoltVirtualPolynomial, TraceDimensions, UnbatchedClaim, UnbatchedClaimExpr, UnbatchedRelation,
};
use crate::SymbolicSumcheck;
use crate::{InputClaims, OutputClaims, SumcheckChallenges};
use jolt_field::Ring;

/// Produced instruction-input virtualization openings (the left/right operand
/// selector flags and their operand values), all sharing the single
/// instruction-input opening point. Generic over the cell.
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
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
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct InstructionInputInputClaims<C> {
    #[opening(RightInstructionInput, from = SpartanProductVirtualization)]
    pub right_instruction_input: C,
    #[opening(LeftInstructionInput, from = SpartanProductVirtualization)]
    pub left_instruction_input: C,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct InstructionInputChallenges<F> {
    #[challenge(InstructionInputChallenge::Gamma)]
    pub gamma: F,
}

/// The instruction input-virtualization sumcheck: relates the left/right
/// instruction-input products from the product sumcheck to the per-operand
/// flag/value openings, folded by `gamma` and weighted by the `EqProduct` public.
#[derive(Clone)]
pub struct InputVirtualization {
    shape: TraceDimensions,
}

impl InputVirtualization {
    pub fn unbatched_relation() -> UnbatchedRelation {
        let v = UnbatchedClaimExpr::polynomial;
        UnbatchedRelation {
            output_relation: JoltRelationId::InstructionInputVirtualization,
            gamma: InstructionInputChallenge::Gamma.into(),
            claims: vec![
                UnbatchedClaim {
                    input_relation: JoltRelationId::SpartanProductVirtualization,
                    input: v(JoltVirtualPolynomial::RightInstructionInput),
                    output: v(JoltVirtualPolynomial::InstructionFlags(
                        InstructionFlags::RightOperandIsRs2Value,
                    )) * v(JoltVirtualPolynomial::Rs2Value)
                        + v(JoltVirtualPolynomial::InstructionFlags(
                            InstructionFlags::RightOperandIsImm,
                        )) * v(JoltVirtualPolynomial::Imm),
                    output_weight: InstructionInputPublic::EqProduct.into(),
                    offset: false,
                },
                UnbatchedClaim {
                    input_relation: JoltRelationId::SpartanProductVirtualization,
                    input: v(JoltVirtualPolynomial::LeftInstructionInput),
                    output: v(JoltVirtualPolynomial::InstructionFlags(
                        InstructionFlags::LeftOperandIsRs1Value,
                    )) * v(JoltVirtualPolynomial::Rs1Value)
                        + v(JoltVirtualPolynomial::InstructionFlags(
                            InstructionFlags::LeftOperandIsPC,
                        )) * v(JoltVirtualPolynomial::UnexpandedPC),
                    output_weight: InstructionInputPublic::EqProduct.into(),
                    offset: false,
                },
            ],
        }
    }
}

impl SymbolicSumcheck for InputVirtualization {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = TraceDimensions;
    type Challenges<F> = InstructionInputChallenges<F>;
    type Inputs<C> = InstructionInputInputClaims<C>;
    type Outputs<C> = InstructionInputOutputClaims<C>;

    fn new(shape: TraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::InstructionInputVirtualization
    }

    fn rounds(&self) -> usize {
        self.shape.log_t()
    }

    fn degree(&self) -> usize {
        INPUT_VIRTUALIZATION_DEGREE
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        Self::unbatched_relation().folded_input()
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        Self::unbatched_relation().folded_output()
    }
}
