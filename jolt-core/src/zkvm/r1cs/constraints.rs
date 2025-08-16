use crate::{
    field::JoltField,
    zkvm::{
        instruction::CircuitFlags,
        r1cs::{
            builder::{ConstraintType, R1CSBuilder},
            ops::{AzType, BzType, CzType},
        },
    },
};

use super::{builder::CombinedUniformBuilder, inputs::JoltR1CSInputs};

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
        // A: OpFlag (boolean)
        // B: LeftInstructionInput (u64) - Rs1Value (u64) -> i65 -> U64AndSign
        // C: 0
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsRs1Value),
            JoltR1CSInputs::LeftInstructionInput,
            JoltR1CSInputs::Rs1Value,
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::U64AndSign,
                c_type: CzType::Zero,
            },
        );

        // if LeftOperandIsPC { assert!(LeftInstructionInput == UnexpandedPC) }
        // A: OpFlag (boolean)
        // B: LeftInstructionInput (u64) - UnexpandedPC (u64) -> i65 -> U64AndSign
        // C: 0
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsPC),
            JoltR1CSInputs::LeftInstructionInput,
            JoltR1CSInputs::UnexpandedPC,
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::U64AndSign,
                c_type: CzType::Zero,
            },
        );

        // if !(LeftOperandIsRs1Value || LeftOperandIsPC)  {
        //     assert!(LeftInstructionInput == 0)
        // }
        // Note that LeftOperandIsRs1Value and LeftOperandIsPC are mutually exclusive flags
        // A: 1 - OpFlag - OpFlag -> [0, 1] -> i8 (mutually exclusive)
        // B: LeftInstructionInput (u64) - 0 -> u64
        // C: 0
        cs.constrain_eq_conditional(
            1 - JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsRs1Value)
                - JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsPC),
            JoltR1CSInputs::LeftInstructionInput,
            0i128,
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::U64,
                c_type: CzType::Zero,
            },
        );

        // if RightOperandIsRs2Value { assert!(RightInstructionInput == Rs2Value) }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsRs2Value),
            JoltR1CSInputs::RightInstructionInput,
            JoltR1CSInputs::Rs2Value,
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::I128,
                c_type: CzType::Zero,
            },
        );

        // if RightOperandIsImm { assert!(RightInstructionInput == Imm) }
        // A: OpFlag (boolean)
        // B: RightInstructionInput (u64) - Imm (i64) -> i128 (can exceed u64::MAX)
        // C: 0
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsImm),
            JoltR1CSInputs::RightInstructionInput,
            JoltR1CSInputs::Imm,
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::I128,
                c_type: CzType::Zero,
            },
        );

        // if !(RightOperandIsRs2Value || RightOperandIsImm)  {
        //     assert!(RightInstructionInput == 0)
        // }
        // Note that RightOperandIsRs2Value and RightOperandIsImm are mutually exclusive flags
        // A: 1 - OpFlag - OpFlag -> [0, 1] -> i8 (mutually exclusive)
        // B: RightInstructionInput (u64) - 0 -> u64
        // C: 0
        cs.constrain_eq_conditional(
            1 - JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsRs2Value)
                - JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsImm),
            JoltR1CSInputs::RightInstructionInput,
            0i128,
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::U64,
                c_type: CzType::Zero,
            },
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
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::I128,
                c_type: CzType::U64,
            },
        );

        // if Load {
        //     assert!(RamReadValue == RamWriteValue)
        // }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::Load),
            JoltR1CSInputs::RamReadValue,
            JoltR1CSInputs::RamWriteValue,
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::I128,
                c_type: CzType::Zero,
            },
        );

        // if Load {
        //     assert!(RamReadValue == RdWriteValue)
        // }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::Load),
            JoltR1CSInputs::RamReadValue,
            JoltR1CSInputs::RdWriteValue,
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::I128,
                c_type: CzType::Zero,
            },
        );

        // if Store {
        //     assert!(Rs2Value == RamWriteValue)
        // }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::Store),
            JoltR1CSInputs::Rs2Value,
            JoltR1CSInputs::RamWriteValue,
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::I128,
                c_type: CzType::Zero,
            },
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
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::I128,
                c_type: CzType::U64, // TODO(should be U64AndSign)
            },
        );

        // If AddOperands {
        //     assert!(RightLookupOperand == LeftInstructionInput + RightInstructionInput)
        // }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands),
            JoltR1CSInputs::RightLookupOperand,
            JoltR1CSInputs::LeftInstructionInput + JoltR1CSInputs::RightInstructionInput,
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::U128AndSign,
                c_type: CzType::Zero,
            },
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
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::U128AndSign,
                c_type: CzType::Zero,
            },
        );

        // if MultiplyOperands {
        //     assert!(RightLookupOperand == Rs1Value * Rs2Value)
        // }
        cs.constrain_prod(
            JoltR1CSInputs::RightInstructionInput,
            JoltR1CSInputs::LeftInstructionInput,
            JoltR1CSInputs::Product,
            ConstraintType {
                a_type: AzType::U64,
                b_type: BzType::U64,
                c_type: CzType::U128,
            },
        );
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands),
            JoltR1CSInputs::RightLookupOperand,
            JoltR1CSInputs::Product,
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::U128AndSign,
                c_type: CzType::Zero,
            },
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
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::U128AndSign,
                c_type: CzType::Zero,
            },
        );

        // if Assert {
        //     assert!(LookupOutput == 1)
        // }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::Assert),
            JoltR1CSInputs::LookupOutput,
            1i128,
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::U64AndSign,
                c_type: CzType::Zero,
            },
        );

        // if Rd != 0 && WriteLookupOutputToRD {
        //     assert!(RdWriteValue == LookupOutput)
        // }
        cs.constrain_prod(
            JoltR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToRD),
            JoltR1CSInputs::Rd,
            JoltR1CSInputs::WriteLookupOutputToRD,
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::I8,
                c_type: CzType::I8,
            },
        );
        cs.constrain_eq_conditional(
            JoltR1CSInputs::WriteLookupOutputToRD,
            JoltR1CSInputs::RdWriteValue,
            JoltR1CSInputs::LookupOutput,
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::U64AndSign,
                c_type: CzType::Zero,
            },
        );

        // if Rd != 0 && Jump {
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
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::I8,
                c_type: CzType::I8,
            },
        );
        cs.constrain_eq_conditional(
            JoltR1CSInputs::WritePCtoRD,
            JoltR1CSInputs::RdWriteValue,
            JoltR1CSInputs::UnexpandedPC + 4i128
                - 2 * JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed),
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::I128,
                c_type: CzType::Zero,
            },
        );

        // if Jump && !NextIsNoop {
        //     assert!(NextUnexpandedPC == LookupOutput)
        // }
        cs.constrain_prod(
            JoltR1CSInputs::OpFlags(CircuitFlags::Jump),
            1 - JoltR1CSInputs::NextIsNoop,
            JoltR1CSInputs::ShouldJump,
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::I8,
                c_type: CzType::I8,
            },
        );
        cs.constrain_eq_conditional(
            JoltR1CSInputs::ShouldJump,
            JoltR1CSInputs::NextUnexpandedPC,
            JoltR1CSInputs::LookupOutput,
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::U64AndSign,
                c_type: CzType::Zero,
            },
        );

        // if Branch && LookupOutput {
        //     assert!(NextUnexpandedPC == UnexpandedPC + Imm)
        // }
        cs.constrain_prod(
            JoltR1CSInputs::OpFlags(CircuitFlags::Branch),
            JoltR1CSInputs::LookupOutput,
            JoltR1CSInputs::ShouldBranch,
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::U64,
                c_type: CzType::U64,
            },
        );
        cs.constrain_eq_conditional(
            JoltR1CSInputs::ShouldBranch,
            JoltR1CSInputs::NextUnexpandedPC,
            JoltR1CSInputs::UnexpandedPC + JoltR1CSInputs::Imm,
            ConstraintType {
                a_type: AzType::U64,
                b_type: BzType::I128,
                c_type: CzType::Zero,
            },
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
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::I8,
                c_type: CzType::I8,
            },
        );
        cs.constrain_eq_conditional(
            1 - JoltR1CSInputs::ShouldBranch - JoltR1CSInputs::OpFlags(CircuitFlags::Jump),
            JoltR1CSInputs::NextUnexpandedPC,
            JoltR1CSInputs::UnexpandedPC + 4i128
                - 4 * JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC)
                - 2 * JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed)
                + 2 * JoltR1CSInputs::CompressedDoNotUpdateUnexpPC,
            ConstraintType {
                a_type: AzType::I128,
                b_type: BzType::I128,
                c_type: CzType::Zero,
            },
        );

        // if Inline {
        //     assert!(NextPC == PC + 1)
        // }
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::InlineSequenceInstruction),
            JoltR1CSInputs::NextPC,
            JoltR1CSInputs::PC + 1i128,
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::I128,
                c_type: CzType::Zero,
            },
        );
    }
}