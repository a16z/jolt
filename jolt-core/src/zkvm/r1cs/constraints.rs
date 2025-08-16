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
        // A: OpFlag (boolean)
        // B: RightInstructionInput (i64) - Rs2Value (u64) -> i128
        // C: 0
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
        // B: RightInstructionInput (i64) - Imm (i64) -> i128
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
        // B: RightInstructionInput (i64) - 0 -> i128 (just to be safe)
        // C: 0
        cs.constrain_eq_conditional(
            1 - JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsRs2Value)
                - JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsImm),
            JoltR1CSInputs::RightInstructionInput,
            0i128,
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::I128,
                c_type: CzType::Zero,
            },
        );

        // if Load || Store {
        //     assert!(RamAddress == Rs1Value + Imm)
        // } else {
        //     assert!(RamAddress == 0)
        // }
        // A: OpFlag + OpFlag -> [0, 2] -> i8
        // B: (Rs1Value(u64) + Imm(i64)) - 0 -> i65 -> i128
        // C: RamAddress (u64) - 0 -> u64
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
        // A: OpFlag (boolean)
        // B: RamReadValue (u64) - RamWriteValue (u64) -> i65 -> U64AndSign
        // C: 0
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::Load),
            JoltR1CSInputs::RamReadValue,
            JoltR1CSInputs::RamWriteValue,
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::U64AndSign,
                c_type: CzType::Zero,
            },
        );

        // if Load {
        //     assert!(RamReadValue == RdWriteValue)
        // }
        // A: OpFlag (boolean)
        // B: RamReadValue (u64) - RdWriteValue (u64) -> i65 -> U64AndSign
        // C: 0
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::Load),
            JoltR1CSInputs::RamReadValue,
            JoltR1CSInputs::RdWriteValue,
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::U64AndSign,
                c_type: CzType::Zero,
            },
        );

        // if Store {
        //     assert!(Rs2Value == RamWriteValue)
        // }
        // A: OpFlag (boolean)
        // B: Rs2Value (u64) - RamWriteValue (u64) -> i65 -> U64AndSign
        // C: 0
        cs.constrain_eq_conditional(
            JoltR1CSInputs::OpFlags(CircuitFlags::Store),
            JoltR1CSInputs::Rs2Value,
            JoltR1CSInputs::RamWriteValue,
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::U64AndSign,
                c_type: CzType::Zero,
            },
        );

        // if AddOperands || SubtractOperands || MultiplyOperands {
        //     // Lookup query is just RightLookupOperand
        //     assert!(LeftLookupOperand == 0)
        // } else {
        //     assert!(LeftLookupOperand == LeftInstructionInput)
        // }
        // A: OpFlag + OpFlag + OpFlag -> [0, 3] -> i8
        // B: 0 - LeftInstructionInput (u64) -> i65 -> U64AndSign
        // C: LeftLookupOperand (u64) - LeftInstructionInput (u64) -> i65 -> U64AndSign
        cs.constrain_if_else(
            JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands)
                + JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands)
                + JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands),
            0i128,
            JoltR1CSInputs::LeftInstructionInput,
            JoltR1CSInputs::LeftLookupOperand,
            ConstraintType {
                a_type: AzType::I8,
                b_type: BzType::U64AndSign,
                c_type: CzType::U64AndSign,
            },
        );

        // If AddOperands {
        //     assert!(RightLookupOperand == LeftInstructionInput + RightInstructionInput)
        // }
        // A: OpFlag (boolean)
        // B: RightLookupOperand (u128) - (L(u64) + R(i64)) -> i129 -> U128AndSign
        // C: 0
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
        // A: OpFlag (boolean)
        // B: RightLookupOperand (u128) - (L(u64) - R(i64) + 2^64) -> i129 -> U128AndSign
        // C: 0
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
        // Constraint 1: Product = L * R
        // A: LeftInstructionInput (u64)
        // B: RightInstructionInput (i64)
        // C: Product (u128)
        cs.constrain_prod(
            JoltR1CSInputs::LeftInstructionInput,
            JoltR1CSInputs::RightInstructionInput,
            JoltR1CSInputs::Product,
            ConstraintType {
                a_type: AzType::U64,
                b_type: BzType::U64AndSign,
                c_type: CzType::U128AndSign,
            },
        );
        // Constraint 2: if Mul { RLookup == Product }
        // A: OpFlag (boolean)
        // B: RightLookupOperand (u128) - Product (u128) -> i129 -> U128AndSign
        // C: 0
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
        // A: 1 - Flags... -> [-3, 1] -> i8
        // B: RightLookupOperand (u128) - RightInstructionInput (i64) -> i129 -> U128AndSign
        // C: 0
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
        // A: OpFlag (boolean)
        // B: LookupOutput (u64) - 1 -> i65 -> U64AndSign (if LookupOutput=0, result is -1)
        // C: 0
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
        // Constraint 1: WriteLookupOutputToRD = OpFlag * Rd
        // A: OpFlag (boolean)
        // B: Rd (u5) -> i8
        // C: WriteLookupOutputToRD (u5) -> i8
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
        // Constraint 2: if WriteLookupOutputToRD { RdWriteValue == LookupOutput }
        // A: WriteLookupOutputToRD (u5) -> i8
        // B: RdWriteValue (u64) - LookupOutput (u64) -> i65 -> U64AndSign
        // C: 0
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
        // Constraint 1: WritePCtoRD = OpFlag * Rd
        // A: OpFlag (boolean)
        // B: Rd (u5) -> i8
        // C: WritePCtoRD (u5) -> i8
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
        // Constraint 2: if WritePCtoRD { RdWriteValue == PC + ... }
        // A: WritePCtoRD (u5) -> i8
        // B: RdWriteValue(u64) - (UnexpandedPC(u64) + const) -> i65 -> i128
        // C: 0
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
        // Constraint 1: ShouldJump = OpFlag * (1 - OpFlag)
        // A: OpFlag (boolean)
        // B: 1 - OpFlag (boolean) -> i8
        // C: ShouldJump (boolean) -> i8
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
        // Constraint 2: if ShouldJump { NextUnexpandedPC == LookupOutput }
        // A: ShouldJump (boolean) -> i8
        // B: NextUnexpandedPC (u64) - LookupOutput (u64) -> i65 -> U64AndSign
        // C: 0
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
        // Constraint 1: ShouldBranch = OpFlag * LookupOutput
        // A: OpFlag (boolean)
        // B: LookupOutput (u64)
        // C: ShouldBranch (u64)
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
        // Constraint 2: if ShouldBranch { NextUnexpandedPC == UnexpandedPC + Imm }
        // A: ShouldBranch (u64)
        // B: NextUnexpandedPC(u64) - (UnexpandedPC(u64) + Imm(i64)) -> i65 -> i128
        // C: 0
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
        // Constraint 1: CompressedDoNotUpdateUnexpPC = isCompressed * DoNotUpdateUnexpandedPC
        // A: OpFlag (boolean)
        // B: OpFlag (boolean)
        // C: CompressedDoNotUpdateUnexpPC (boolean) -> i8
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
        // Constraint 2: if !(ShouldBranch || Jump) { NextUnexpandedPC == PC + ... }
        // A: 1 - ShouldBranch(u64) - Jump(flag) -> i64 -> U64AndSign
        // B: NextUnexpandedPC(u64) - (UnexpandedPC(u64) + consts) -> i65 -> i128
        // C: 0
        cs.constrain_eq_conditional(
            1 - JoltR1CSInputs::ShouldBranch - JoltR1CSInputs::OpFlags(CircuitFlags::Jump),
            JoltR1CSInputs::NextUnexpandedPC,
            JoltR1CSInputs::UnexpandedPC + 4i128
                - 4 * JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC)
                - 2 * JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed)
                + 2 * JoltR1CSInputs::CompressedDoNotUpdateUnexpPC,
            ConstraintType {
                a_type: AzType::U64AndSign,
                b_type: BzType::I128,
                c_type: CzType::Zero,
            },
        );

        // if Inline {
        //     assert!(NextPC == PC + 1)
        // }
        // A: OpFlag (boolean)
        // B: NextPC(u64) - (PC(u64) + 1) -> i65 -> i128
        // C: 0
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
