//! Instruction input-virtualization symbolic sumcheck relation.

use jolt_riscv::InstructionFlags;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::instruction::{
    imm, left_instruction_input_product, left_operand_is_pc, left_operand_is_rs1,
    right_instruction_input_product, right_operand_is_imm, right_operand_is_rs2, rs1_value,
    rs2_value, unexpanded_pc, INPUT_VIRTUALIZATION_DEGREE,
};
use crate::protocols::jolt::{
    InstructionInputChallenge, InstructionInputPublic, JoltExpr, JoltRelationId, TraceDimensions,
};
use crate::SymbolicSumcheck;
use crate::{challenge, derived, opening, InputClaims, OutputClaims, SumcheckChallenges};
use jolt_field::RingCore;

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

/// Fiat-Shamir challenge drawn by the instruction input-virtualization sumcheck.
#[derive(Clone, Copy, Debug, SumcheckChallenges)]
pub struct InstructionInputChallenges<F> {
    #[challenge(InstructionInputChallenge::Gamma)]
    pub gamma: F,
}

/// The instruction input-virtualization sumcheck: relates the left/right
/// instruction-input products from the product sumcheck to the per-operand
/// flag/value openings, folded by `gamma` and weighted by the `EqProduct` public.
pub struct InputVirtualization {
    shape: TraceDimensions,
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

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(right_instruction_input_product())
            + challenge(InstructionInputChallenge::Gamma)
                * opening(left_instruction_input_product())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        derived(InstructionInputPublic::EqProduct)
            * opening(right_operand_is_rs2())
            * opening(rs2_value())
            + derived(InstructionInputPublic::EqProduct)
                * opening(right_operand_is_imm())
                * opening(imm())
            + derived(InstructionInputPublic::EqProduct)
                * challenge(InstructionInputChallenge::Gamma)
                * opening(left_operand_is_rs1())
                * opening(rs1_value())
            + derived(InstructionInputPublic::EqProduct)
                * challenge(InstructionInputChallenge::Gamma)
                * opening(left_operand_is_pc())
                * opening(unexpanded_pc())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::{JoltChallengeId, JoltDerivedId};
    use jolt_field::{Fr, FromPrimitiveInt};

    fn trace_dimensions() -> TraceDimensions {
        TraceDimensions::new(5)
    }

    #[test]
    fn input_virtualization_evaluates_like_core_formula() {
        let relation = InputVirtualization::new(trace_dimensions());

        let right_input = Fr::from_u64(3);
        let left_input = Fr::from_u64(5);
        let right_is_rs2 = Fr::from_u64(7);
        let rs2 = Fr::from_u64(11);
        let right_is_imm = Fr::from_u64(13);
        let imm_value = Fr::from_u64(17);
        let left_is_rs1 = Fr::from_u64(19);
        let rs1 = Fr::from_u64(23);
        let left_is_pc = Fr::from_u64(29);
        let pc = Fr::from_u64(31);
        let gamma = Fr::from_u64(37);
        let eq_product = Fr::from_u64(41);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == right_instruction_input_product() => right_input,
                id if id == left_instruction_input_product() => left_input,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::InstructionInput(InstructionInputChallenge::Gamma) => gamma,
                _ => zero,
            },
            |_| zero,
        );

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == right_operand_is_rs2() => right_is_rs2,
                id if id == rs2_value() => rs2,
                id if id == right_operand_is_imm() => right_is_imm,
                id if id == imm() => imm_value,
                id if id == left_operand_is_rs1() => left_is_rs1,
                id if id == rs1_value() => rs1,
                id if id == left_operand_is_pc() => left_is_pc,
                id if id == unexpanded_pc() => pc,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::InstructionInput(InstructionInputChallenge::Gamma) => gamma,
                _ => zero,
            },
            |id| match *id {
                JoltDerivedId::InstructionInput(InstructionInputPublic::EqProduct) => eq_product,
                _ => zero,
            },
        );

        assert_eq!(input, right_input + gamma * left_input);
        assert_eq!(
            output,
            eq_product
                * (right_is_rs2 * rs2
                    + right_is_imm * imm_value
                    + gamma * left_is_rs1 * rs1
                    + gamma * left_is_pc * pc)
        );
    }

    #[test]
    fn input_virtualization_symbolic_matches_dependencies() {
        let relation = InputVirtualization::new(TraceDimensions::new(5));
        assert_eq!(
            InputVirtualization::id(),
            JoltRelationId::InstructionInputVirtualization
        );
        assert_eq!(relation.rounds(), TraceDimensions::new(5).log_t());
        assert_eq!(relation.degree(), INPUT_VIRTUALIZATION_DEGREE);
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![JoltChallengeId::from(InstructionInputChallenge::Gamma)]
        );
        assert_eq!(
            relation.required_deriveds::<Fr>(),
            vec![JoltDerivedId::from(InstructionInputPublic::EqProduct)]
        );
    }
}
