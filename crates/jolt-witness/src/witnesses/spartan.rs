use jolt_claims::protocols::jolt::geometry::spartan::SPARTAN_OUTER_R1CS_INPUTS;
use jolt_claims::protocols::jolt::JoltPolynomialId;
use jolt_field::signed::{S128, S64};
use jolt_riscv::JoltTraceRow as TraceRow;
use jolt_riscv::{CircuitFlags, InstructionFlags};

use super::{
    lookup_values, row_is_noop, Imm, InstructionFlag, LeftInstructionInput, LeftLookupOperand,
    LookupOutput, NextIsFirstInSequence, NextIsNoop, NextIsVirtual, NextPc, NextUnexpandedPc,
    OpFlag, Pc, Product, RamAddress, RamReadValue, RamWriteValue, RdWriteValue,
    RightInstructionInput, RightLookupOperand, Rs1Value, Rs2Value, ShouldBranch, ShouldJump,
    UnexpandedPc, WitnessEnv,
};
use crate::{WitnessBundle, WitnessError};

/// The canonical Spartan outer inputs extracted in one pass over a cycle.
#[derive(Clone, Copy, Debug)]
pub struct SpartanOuterRow {
    pub left_instruction_input: LeftInstructionInput,
    pub right_instruction_input: RightInstructionInput,
    pub product: Product,
    pub should_branch: ShouldBranch,
    pub pc: Pc,
    pub unexpanded_pc: UnexpandedPc,
    pub imm: Imm,
    pub ram_address: RamAddress,
    pub rs1_value: Rs1Value,
    pub rs2_value: Rs2Value,
    pub rd_write_value: RdWriteValue,
    pub ram_read_value: RamReadValue,
    pub ram_write_value: RamWriteValue,
    pub left_lookup_operand: LeftLookupOperand,
    pub right_lookup_operand: RightLookupOperand,
    pub next_unexpanded_pc: NextUnexpandedPc,
    pub next_pc: NextPc,
    pub next_is_virtual: NextIsVirtual,
    pub next_is_first_in_sequence: NextIsFirstInSequence,
    pub lookup_output: LookupOutput,
    pub should_jump: ShouldJump,
    pub branch_flag: InstructionFlag,
    pub is_noop: InstructionFlag,
    pub next_is_noop: NextIsNoop,
    pub add_operands: OpFlag,
    pub subtract_operands: OpFlag,
    pub multiply_operands: OpFlag,
    pub load: OpFlag,
    pub store: OpFlag,
    pub jump: OpFlag,
    pub write_lookup_output_to_rd: OpFlag,
    pub virtual_instruction: OpFlag,
    pub assert_flag: OpFlag,
    pub do_not_update_unexpanded_pc: OpFlag,
    pub advice: OpFlag,
    pub is_compressed: OpFlag,
    pub is_first_in_sequence: OpFlag,
    pub is_last_in_sequence: OpFlag,
    pub left_operand_is_rs1: InstructionFlag,
    pub left_operand_is_pc: InstructionFlag,
    pub right_operand_is_rs2: InstructionFlag,
    pub right_operand_is_imm: InstructionFlag,
}

impl WitnessBundle for SpartanOuterRow {
    #[inline]
    fn from_row(
        row: &TraceRow,
        next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let circuit_flags = row.circuit_flags();
        let instruction_flags = row.instruction_flags();
        let (
            (left_instruction_input, right_instruction_input),
            (left_lookup_operand, right_lookup_operand),
            lookup_output,
        ) = lookup_values(row);
        let next_flags = next.map(TraceRow::circuit_flags);
        let flag = |flag| OpFlag(circuit_flags[flag]);

        Ok(Self {
            left_instruction_input: LeftInstructionInput(left_instruction_input),
            right_instruction_input: RightInstructionInput(right_instruction_input),
            product: Product(
                S64::from_u64(left_instruction_input)
                    .mul_trunc::<2, 2>(&S128::from_i128(right_instruction_input)),
            ),
            should_branch: ShouldBranch(
                instruction_flags[InstructionFlags::Branch] && lookup_output == 1,
            ),
            pc: Pc(row.pc()),
            unexpanded_pc: UnexpandedPc(row.unexpanded_pc()),
            imm: Imm(row.imm()),
            ram_address: RamAddress(row.ram_address()),
            rs1_value: Rs1Value(row.rs1_value()),
            rs2_value: Rs2Value(row.rs2_value()),
            rd_write_value: RdWriteValue(row.rd_write_value()),
            ram_read_value: RamReadValue(row.ram_read_value()),
            ram_write_value: RamWriteValue(row.ram_write_value()),
            left_lookup_operand: LeftLookupOperand(left_lookup_operand),
            right_lookup_operand: RightLookupOperand(right_lookup_operand),
            next_unexpanded_pc: NextUnexpandedPc(next.map_or(0, TraceRow::unexpanded_pc)),
            next_pc: NextPc(next.map_or(0, TraceRow::pc)),
            next_is_virtual: NextIsVirtual(
                next_flags.is_some_and(|flags| flags[CircuitFlags::VirtualInstruction]),
            ),
            next_is_first_in_sequence: NextIsFirstInSequence(
                next_flags.is_some_and(|flags| flags[CircuitFlags::IsFirstInSequence]),
            ),
            lookup_output: LookupOutput(lookup_output),
            should_jump: ShouldJump(
                circuit_flags[CircuitFlags::Jump] && !next.is_some_and(row_is_noop),
            ),
            branch_flag: InstructionFlag(instruction_flags[InstructionFlags::Branch]),
            is_noop: InstructionFlag(instruction_flags[InstructionFlags::IsNoop]),
            next_is_noop: NextIsNoop(next.is_none_or(row_is_noop)),
            add_operands: flag(CircuitFlags::AddOperands),
            subtract_operands: flag(CircuitFlags::SubtractOperands),
            multiply_operands: flag(CircuitFlags::MultiplyOperands),
            load: flag(CircuitFlags::Load),
            store: flag(CircuitFlags::Store),
            jump: flag(CircuitFlags::Jump),
            write_lookup_output_to_rd: flag(CircuitFlags::WriteLookupOutputToRD),
            virtual_instruction: flag(CircuitFlags::VirtualInstruction),
            assert_flag: flag(CircuitFlags::Assert),
            do_not_update_unexpanded_pc: flag(CircuitFlags::DoNotUpdateUnexpandedPC),
            advice: flag(CircuitFlags::Advice),
            is_compressed: flag(CircuitFlags::IsCompressed),
            is_first_in_sequence: flag(CircuitFlags::IsFirstInSequence),
            is_last_in_sequence: flag(CircuitFlags::IsLastInSequence),
            left_operand_is_rs1: InstructionFlag(
                instruction_flags[InstructionFlags::LeftOperandIsRs1Value],
            ),
            left_operand_is_pc: InstructionFlag(
                instruction_flags[InstructionFlags::LeftOperandIsPC],
            ),
            right_operand_is_rs2: InstructionFlag(
                instruction_flags[InstructionFlags::RightOperandIsRs2Value],
            ),
            right_operand_is_imm: InstructionFlag(
                instruction_flags[InstructionFlags::RightOperandIsImm],
            ),
        })
    }

    fn annotated_ids() -> Vec<JoltPolynomialId> {
        SPARTAN_OUTER_R1CS_INPUTS
            .into_iter()
            .map(JoltPolynomialId::Virtual)
            .collect()
    }
}
