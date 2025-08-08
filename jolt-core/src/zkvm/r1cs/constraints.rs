use crate::{field::JoltField, zkvm::instruction::CircuitFlags};

use super::{
    builder::{CombinedUniformBuilder, R1CSBuilder},
    inputs::JoltR1CSInputs,
};

pub const PC_START_ADDRESS: i64 = 0x80000000;

pub trait R1CSConstraints<F: JoltField> {
    /// Constructs the R1CS constraints for Jolt. These constraints must hold for every step of
    /// the execution trace.
    fn construct_constraints(padded_trace_length: usize) -> CombinedUniformBuilder<F> {
        let mut r1cs_builder = R1CSBuilder::new();
        Self::normal_constraints(&mut r1cs_builder);
        Self::may_overflow_constraints(&mut r1cs_builder);

        CombinedUniformBuilder::construct(r1cs_builder, padded_trace_length)
    }

    /// Most of Jolt's constraints are "normal", meaning their Az/Bz/Cz values will not
    /// overflow a `i128`.
    fn normal_constraints(builder: &mut R1CSBuilder);

    /// There are some constraints that may involve arithmetic operations that can overflow a `i128`.
    /// We collect them here for special handling.
    /// TODO: distinguish between overflow on Az/Bz and overflow on Cz. The latter is actually
    /// not a problem in the small value rounds.
    fn may_overflow_constraints(builder: &mut R1CSBuilder);
}

pub struct JoltRV32IMConstraints;
impl<F: JoltField> R1CSConstraints<F> for JoltRV32IMConstraints {
    fn normal_constraints(cs: &mut R1CSBuilder) {
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
            0i128,
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
            0i128,
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
            0i128,
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
            0i128,
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
                + (0xffffffffffffffffi128 + 1),
        );

        // Two constraints that can overflow are moved to `may_overflow_constraints`
        // 1. Product constraint: assert(Product = RightInstructionInput * LeftInstructionInput)
        // 2. Conditional constraint: if MultiplyOperands is true, assert(RightLookupOperand = Product)

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
            1i128,
        );

        // if WriteLookupOutputToRD && Rd != 0 {
        //     assert!(RdWriteValue == LookupOutput)
        // }
        cs.constrain_prod(
            JoltR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToRD),
            JoltR1CSInputs::Rd,
            JoltR1CSInputs::WriteLookupOutputToRD,
        );
        cs.constrain_eq_conditional(
            JoltR1CSInputs::WriteLookupOutputToRD,
            JoltR1CSInputs::RdWriteValue,
            JoltR1CSInputs::LookupOutput,
        );

        // if Jump && Rd != 0 {
        //     if !isCompressed {
        //          assert!(RdWriteValue == UnexpandedPC + 4)
        //     } else {
        //          assert!(RdWriteValue == UnexpandedPC + 2)
        //     }
        // }
        cs.constrain_prod(
            JoltR1CSInputs::OpFlags(CircuitFlags::Jump),
            JoltR1CSInputs::Rd,
            JoltR1CSInputs::WritePCtoRD,
        );
        cs.constrain_eq_conditional(
            JoltR1CSInputs::WritePCtoRD,
            JoltR1CSInputs::RdWriteValue,
            JoltR1CSInputs::UnexpandedPC + 4i128
                - 2 * JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed),
        );

        // if Jump && !NextIsNoop {
        //     assert!(NextUnexpandedPC == LookupOutput)
        // }
        cs.constrain_prod(
            JoltR1CSInputs::OpFlags(CircuitFlags::Jump),
            1 - JoltR1CSInputs::NextIsNoop,
            JoltR1CSInputs::ShouldJump,
        );
        cs.constrain_eq_conditional(
            JoltR1CSInputs::ShouldJump,
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
        //     } else if isCompressed {
        //         assert!(NextUnexpandedPC == UnexpandedPC + 2)
        //     } else {
        //         assert!(NextUnexpandedPC == UnexpandedPC + 4)
        //     }
        // }
        // Note that ShouldBranch and Jump instructions are mutually exclusive
        cs.constrain_prod(
            JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed),
            JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC),
            JoltR1CSInputs::CompressedDoNotUpdateUnexpPC,
        );
        cs.constrain_eq_conditional(
            1 - JoltR1CSInputs::ShouldBranch - JoltR1CSInputs::OpFlags(CircuitFlags::Jump),
            JoltR1CSInputs::NextUnexpandedPC,
            JoltR1CSInputs::UnexpandedPC + 4i128
                - 4 * JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC)
                - 2 * JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed)
                + 2 * JoltR1CSInputs::CompressedDoNotUpdateUnexpPC,
        );

        // if Inline {
        //     assert!(NextPC == PC + 1)
        // }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::InlineSequenceInstruction),
            JoltR1CSInputs::NextPC,
            JoltR1CSInputs::PC + 1i128,
        );
    }

    /// Among the uniform constraints, there are some that may involve arithmetic operations that
    /// can overflow a `i128`. We collect them here for special handling.
    fn may_overflow_constraints(cs: &mut R1CSBuilder) {
        // assert!(Product == RightInstructionInput * LeftInstructionInput)
        // This constraint only overflows on Cz, as it is u128 instead of i128. This can be handled.
        cs.constrain_prod(
            JoltR1CSInputs::RightInstructionInput,
            JoltR1CSInputs::LeftInstructionInput,
            JoltR1CSInputs::Product,
        );

        // if MultiplyOperands {
        //     assert!(RightLookupOperand == Product)
        // }
        // This constraint only overflows on Bz, and since it is `u128 - u64`, it only fits in `i129`.
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands),
            JoltR1CSInputs::RightLookupOperand,
            JoltR1CSInputs::Product,
        );
    }
}
