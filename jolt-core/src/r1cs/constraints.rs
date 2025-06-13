use crate::{field::JoltField, jolt::instruction::CircuitFlags};

use super::{
    builder::{CombinedUniformBuilder, R1CSBuilder},
    inputs::JoltR1CSInputs,
};

pub const PC_START_ADDRESS: i64 = 0x80000000;

pub trait R1CSConstraints<F: JoltField> {
    fn construct_constraints(padded_trace_length: usize) -> CombinedUniformBuilder<F> {
        let mut uniform_builder = R1CSBuilder::new();
        Self::uniform_constraints(&mut uniform_builder);

        CombinedUniformBuilder::construct(uniform_builder, padded_trace_length)
    }
    /// Constructs Jolt's uniform constraints.
    /// Uniform constraints are constraints that hold for each step of
    /// the execution trace.
    fn uniform_constraints(builder: &mut R1CSBuilder);
}

pub struct JoltRV32IMConstraints;
impl<F: JoltField> R1CSConstraints<F> for JoltRV32IMConstraints {
    fn uniform_constraints(cs: &mut R1CSBuilder) {
        // if LeftOperandIsRs1Value { assert!(LeftInstructionInput == Rs1Value) }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsRs1Value),
            JoltR1CSInputs::LeftInstructionInput,
            JoltR1CSInputs::Rs1Value,
        );

        // if LeftOperandIsPC { assert!(LeftInstructionInput == UnexpandedPC) }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsPC),
            JoltR1CSInputs::LeftInstructionInput,
            JoltR1CSInputs::UnexpandedPC,
        );

        // if !(LeftOperandIsRs1Value || LeftOperandIsPC)  {
        //     assert!(LeftInstructionInput == 0)
        // }
        // Note that LeftOperandIsRs1Value and LeftOperandIsPC are mutually exclusive flags
        cs.constrain_eq_conditional(
            1 - JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsRs1Value)
                - JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsPC),
            JoltR1CSInputs::LeftInstructionInput,
            0,
        );

        // if RightOperandIsRs2Value { assert!(RightInstructionInput == Rs2Value) }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsRs2Value),
            JoltR1CSInputs::RightInstructionInput,
            JoltR1CSInputs::Rs2Value,
        );

        // if RightOperandIsImm { assert!(RightInstructionInput == Imm) }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsImm),
            JoltR1CSInputs::RightInstructionInput,
            JoltR1CSInputs::Imm,
        );

        // if !(RightOperandIsRs2Value || RightOperandIsImm)  {
        //     assert!(RightInstructionInput == 0)
        // }
        // Note that RightOperandIsRs2Value and RightOperandIsImm are mutually exclusive flags
        cs.constrain_eq_conditional(
            1 - JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsRs2Value)
                - JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsImm),
            JoltR1CSInputs::RightInstructionInput,
            0,
        );

        // if Load || Store {
        //     assert!(RamAddress == Rs1Value + Imm)
        // } else {
        //     assert!(RamAddress == 0)
        // }
        let is_load_or_store = JoltR1CSInputs::OpFlags(CircuitFlags::Load)
            + JoltR1CSInputs::OpFlags(CircuitFlags::Store);
        cs.constrain_if_else(
            is_load_or_store,
            JoltR1CSInputs::Rs1Value + JoltR1CSInputs::Imm,
            0,
            JoltR1CSInputs::RamAddress,
        );

        // if Load {
        //     assert!(RamReadValue == RamWriteValue)
        // }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::Load),
            JoltR1CSInputs::RamReadValue,
            JoltR1CSInputs::RamWriteValue,
        );

        // if Load {
        //     assert!(RamReadValue == RdWriteValue)
        // }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::Load),
            JoltR1CSInputs::RamReadValue,
            JoltR1CSInputs::RdWriteValue,
        );

        // if Store {
        //     assert!(Rs2Value == RamWriteValue)
        // }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::Store),
            JoltR1CSInputs::Rs2Value,
            JoltR1CSInputs::RamWriteValue,
        );

        // if AddOperands || SubtractOperands || MultiplyOperands {
        //     // Lookup query is just RightLookupOperand
        //     assert!(LeftLookupOperand == 0)
        // } else {
        //     assert!(LeftLookupOperand == LeftInstructionInput)
        // }
        cs.constrain_if_else(
            JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands)
                + JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands)
                + JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands),
            0,
            JoltR1CSInputs::LeftInstructionInput,
            JoltR1CSInputs::LeftLookupOperand,
        );

        // If AddOperands {
        //     assert!(RightLookupOperand == LeftInstructionInput + RightInstructionInput)
        // }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands),
            JoltR1CSInputs::RightLookupOperand,
            JoltR1CSInputs::LeftInstructionInput + JoltR1CSInputs::RightInstructionInput,
        );

        // If SubtractOperands {
        //     assert!(RightLookupOperand == LeftInstructionInput - RightInstructionInput)
        // }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands),
            JoltR1CSInputs::RightLookupOperand,
            // Converts from unsigned to twos-complement representation
            JoltR1CSInputs::LeftInstructionInput - JoltR1CSInputs::RightInstructionInput
                + (0xffffffffi64 + 1),
        );

        // if MultiplyOperands {
        //     assert!(RightLookupOperand == Rs1Value * Rs2Value)
        // }
        cs.constrain_prod(
            JoltR1CSInputs::RightInstructionInput,
            JoltR1CSInputs::LeftInstructionInput,
            JoltR1CSInputs::Product,
        );
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands),
            JoltR1CSInputs::RightLookupOperand,
            JoltR1CSInputs::Product,
        );

        // if !(AddOperands || SubtractOperands || MultiplyOperands || Advice) {
        //     assert!(RightLookupOperand == RightInstructionInput)
        // }
        cs.constrain_eq_conditional(
            1 - JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands)
                - JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands)
                - JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands)
                // Arbitrary untrusted advice goes in right lookup operand
                - JoltR1CSInputs::OpFlags(CircuitFlags::Advice),
            JoltR1CSInputs::RightLookupOperand,
            JoltR1CSInputs::RightInstructionInput,
        );

        // if Assert {
        //     assert!(LookupOutput == 1)
        // }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::Assert),
            JoltR1CSInputs::LookupOutput,
            1,
        );

        // if Rd != 0 && WriteLookupOutputToRD {
        //     assert!(RdWriteValue == LookupOutput)
        // }
        cs.constrain_prod(
            JoltR1CSInputs::Rd,
            JoltR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToRD),
            JoltR1CSInputs::WriteLookupOutputToRD,
        );
        cs.constrain_eq_conditional(
            JoltR1CSInputs::WriteLookupOutputToRD,
            JoltR1CSInputs::RdWriteValue,
            JoltR1CSInputs::LookupOutput,
        );

        // if Rd != 0 && Jump {
        //     assert!(RdWriteValue == UnexpandedPC + 4)
        // }
        cs.constrain_prod(
            JoltR1CSInputs::Rd,
            JoltR1CSInputs::OpFlags(CircuitFlags::Jump),
            JoltR1CSInputs::WritePCtoRD,
        );
        cs.constrain_eq_conditional(
            JoltR1CSInputs::WritePCtoRD,
            JoltR1CSInputs::RdWriteValue,
            JoltR1CSInputs::UnexpandedPC + 4,
        );

        // if Jump {
        //     assert!(NextUnexpandedPC == LookupOutput)
        // }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::Jump),
            JoltR1CSInputs::NextUnexpandedPC,
            JoltR1CSInputs::LookupOutput,
        );

        // if Branch && LookupOutput {
        //     assert!(NextUnexpandedPC == UnexpandedPC + Imm)
        // }
        cs.constrain_prod(
            JoltR1CSInputs::OpFlags(CircuitFlags::Branch),
            JoltR1CSInputs::LookupOutput,
            JoltR1CSInputs::ShouldBranch,
        );
        cs.constrain_eq_conditional(
            JoltR1CSInputs::ShouldBranch,
            JoltR1CSInputs::NextUnexpandedPC,
            JoltR1CSInputs::UnexpandedPC + JoltR1CSInputs::Imm,
        );

        // if !(ShouldBranch || Jump) {
        //     if DoNotUpdatePC {
        //         assert!(NextUnexpandedPC == UnexpandedPC)
        //     } else {
        //         assert!(NextUnexpandedPC == UnexpandedPC + 4)
        //     }
        // }
        // Note that Branch and Jump instructions are mutually exclusive
        cs.constrain_eq_conditional(
            1 - JoltR1CSInputs::ShouldBranch - JoltR1CSInputs::OpFlags(CircuitFlags::Jump),
            JoltR1CSInputs::NextUnexpandedPC,
            JoltR1CSInputs::UnexpandedPC + 4
                - 4 * JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC),
        );

        // if Inline {
        //     assert!(NextPC == PC + 1)
        // }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::InlineSequenceInstruction),
            JoltR1CSInputs::NextPC,
            JoltR1CSInputs::PC + 1,
        );
    }
}
