//! Stage 3 shift and instruction-input rows.

use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JoltVmStage3ShiftRow {
    pub unexpanded_pc: u64,
    pub pc: u64,
    pub is_virtual: bool,
    pub is_first_in_sequence: bool,
    pub is_noop: bool,
}

pub trait JoltVmStage3ShiftRows {
    fn stage3_shift_rows(&self, log_t: usize) -> Result<Vec<JoltVmStage3ShiftRow>, WitnessError>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JoltVmStage3InstructionRegisterRow {
    pub right_operand_is_rs2: bool,
    pub rs2_value: u64,
    pub right_operand_is_imm: bool,
    pub imm: i128,
    pub left_operand_is_rs1: bool,
    pub rs1_value: u64,
    pub left_operand_is_pc: bool,
    pub unexpanded_pc: u64,
    pub rd_write_value: u64,
}

pub trait JoltVmStage3InstructionRegisterRows {
    fn stage3_instruction_register_rows(
        &self,
        log_t: usize,
    ) -> Result<Vec<JoltVmStage3InstructionRegisterRow>, WitnessError>;
}

impl<T: TraceSource + Clone> JoltVmStage3ShiftRows for TraceBackedJoltVmWitness<'_, T> {
    fn stage3_shift_rows(&self, log_t: usize) -> Result<Vec<JoltVmStage3ShiftRow>, WitnessError> {
        let rows = checked_pow2(log_t)?;
        let mut values = Vec::with_capacity(rows);
        let mut trace = self.trace.trace.clone();
        let mut pc_cache = PcLookupCache::default();
        for _ in 0..rows {
            let row = trace.next_row().unwrap_or_default();
            values.push(JoltVmStage3ShiftRow {
                unexpanded_pc: row.instruction.address as u64,
                pc: pc_cache.pc_for_row(&row, self.preprocessing)? as u64,
                is_virtual: row.instruction.virtual_sequence_remaining.is_some(),
                is_first_in_sequence: row.instruction.is_first_in_sequence,
                is_noop: row_is_noop(&row),
            });
        }
        Ok(values)
    }
}

impl<T: TraceSource + Clone> JoltVmStage3InstructionRegisterRows
    for TraceBackedJoltVmWitness<'_, T>
{
    fn stage3_instruction_register_rows(
        &self,
        log_t: usize,
    ) -> Result<Vec<JoltVmStage3InstructionRegisterRow>, WitnessError> {
        let rows = checked_pow2(log_t)?;
        let mut values = Vec::with_capacity(rows);
        let mut trace = self.trace.trace.clone();
        for _ in 0..rows {
            let row = trace.next_row().unwrap_or_default();
            values.push(stage3_instruction_register_row(&row)?);
        }
        Ok(values)
    }
}

pub(crate) fn stage3_instruction_register_row(
    row: &TraceRow,
) -> Result<JoltVmStage3InstructionRegisterRow, WitnessError> {
    let instruction_flags = row_instruction_flags(row)?;
    Ok(JoltVmStage3InstructionRegisterRow {
        right_operand_is_rs2: instruction_flags[InstructionFlags::RightOperandIsRs2Value],
        rs2_value: row.registers.rs2.map_or(0, |read| read.value),
        right_operand_is_imm: instruction_flags[InstructionFlags::RightOperandIsImm],
        imm: row.instruction.operands.imm,
        left_operand_is_rs1: instruction_flags[InstructionFlags::LeftOperandIsRs1Value],
        rs1_value: row.registers.rs1.map_or(0, |read| read.value),
        left_operand_is_pc: instruction_flags[InstructionFlags::LeftOperandIsPC],
        unexpanded_pc: row.instruction.address as u64,
        rd_write_value: row.registers.rd.map_or(0, |write| write.post_value),
    })
}
