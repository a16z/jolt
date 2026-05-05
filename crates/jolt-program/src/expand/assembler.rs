use common::constants::{RISCV_REGISTER_COUNT, VIRTUAL_INSTRUCTION_RESERVED_REGISTER_COUNT};
use jolt_riscv::{InstructionKind, NormalizedInstruction, NormalizedOperands};

use crate::expand::{allocator::ExpansionAllocator, expand_instruction, metadata, ExpansionError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Value {
    Imm(u64),
    Reg(u8),
}

#[derive(Debug)]
pub struct InstrAssembler<'a> {
    address: usize,
    is_compressed: bool,
    has_inline_instr_format: bool,
    sequence: Vec<NormalizedInstruction>,
    allocator: &'a mut ExpansionAllocator,
}

impl<'a> InstrAssembler<'a> {
    pub fn new(address: usize, is_compressed: bool, allocator: &'a mut ExpansionAllocator) -> Self {
        Self {
            address,
            is_compressed,
            has_inline_instr_format: false,
            sequence: Vec::new(),
            allocator,
        }
    }

    pub fn new_inline(
        address: usize,
        is_compressed: bool,
        allocator: &'a mut ExpansionAllocator,
    ) -> Self {
        Self {
            address,
            is_compressed,
            has_inline_instr_format: true,
            sequence: Vec::new(),
            allocator,
        }
    }

    pub fn allocator(&mut self) -> &mut ExpansionAllocator {
        self.allocator
    }

    pub fn emit(
        &mut self,
        instruction_kind: InstructionKind,
        operands: NormalizedOperands,
    ) -> Result<(), ExpansionError> {
        if self.has_inline_instr_format {
            Self::validate_inline_write_target(operands.rd)?;
        }
        let instruction = NormalizedInstruction {
            instruction_kind,
            address: self.address,
            operands,
            virtual_sequence_remaining: Some(0),
            is_first_in_sequence: false,
            is_compressed: false,
        };
        self.sequence
            .extend(expand_instruction(&instruction, self.allocator)?);
        Ok(())
    }

    pub fn emit_r(
        &mut self,
        instruction_kind: InstructionKind,
        rd: u8,
        rs1: u8,
        rs2: u8,
    ) -> Result<(), ExpansionError> {
        self.emit(
            instruction_kind,
            NormalizedOperands {
                rd: Some(rd),
                rs1: Some(rs1),
                rs2: Some(rs2),
                imm: 0,
            },
        )
    }

    pub fn emit_i(
        &mut self,
        instruction_kind: InstructionKind,
        rd: u8,
        rs1: u8,
        imm: i128,
    ) -> Result<(), ExpansionError> {
        self.emit(
            instruction_kind,
            NormalizedOperands {
                rd: Some(rd),
                rs1: Some(rs1),
                rs2: None,
                imm,
            },
        )
    }

    pub fn emit_s(
        &mut self,
        instruction_kind: InstructionKind,
        rs1: u8,
        rs2: u8,
        imm: i128,
    ) -> Result<(), ExpansionError> {
        self.emit(
            instruction_kind,
            NormalizedOperands {
                rd: None,
                rs1: Some(rs1),
                rs2: Some(rs2),
                imm,
            },
        )
    }

    pub fn emit_b(
        &mut self,
        instruction_kind: InstructionKind,
        rs1: u8,
        rs2: u8,
        imm: i128,
    ) -> Result<(), ExpansionError> {
        self.emit(
            instruction_kind,
            NormalizedOperands {
                rd: None,
                rs1: Some(rs1),
                rs2: Some(rs2),
                imm,
            },
        )
    }

    pub fn emit_j(
        &mut self,
        instruction_kind: InstructionKind,
        rd: u8,
        imm: i128,
    ) -> Result<(), ExpansionError> {
        self.emit(
            instruction_kind,
            NormalizedOperands {
                rd: Some(rd),
                rs1: None,
                rs2: None,
                imm,
            },
        )
    }

    pub fn emit_u(
        &mut self,
        instruction_kind: InstructionKind,
        rd: u8,
        imm: i128,
    ) -> Result<(), ExpansionError> {
        self.emit(
            instruction_kind,
            NormalizedOperands {
                rd: Some(rd),
                rs1: None,
                rs2: None,
                imm,
            },
        )
    }

    pub fn emit_align(
        &mut self,
        instruction_kind: InstructionKind,
        rs1: u8,
        imm: i128,
    ) -> Result<(), ExpansionError> {
        self.emit(
            instruction_kind,
            NormalizedOperands {
                rd: None,
                rs1: Some(rs1),
                rs2: None,
                imm,
            },
        )
    }

    pub fn finalize(mut self) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
        finalize_sequence(&mut self.sequence, self.is_compressed)?;
        Ok(self.sequence)
    }

    pub fn finalize_inline(mut self) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
        for register in self.allocator.take_registers_for_reset()? {
            self.emit_i(InstructionKind::ADDI, register, 0, 0)?;
        }
        self.finalize()
    }

    fn validate_inline_write_target(rd: Option<u8>) -> Result<(), ExpansionError> {
        let Some(register) = rd else {
            return Ok(());
        };
        let minimum_register = RISCV_REGISTER_COUNT + VIRTUAL_INSTRUCTION_RESERVED_REGISTER_COUNT;
        if register == 0 || register >= minimum_register {
            return Ok(());
        }
        Err(ExpansionError::InvalidInlineWriteTarget {
            register,
            minimum_register,
        })
    }
}

fn finalize_sequence(
    sequence: &mut [NormalizedInstruction],
    is_compressed: bool,
) -> Result<(), ExpansionError> {
    if sequence.is_empty() {
        return Err(ExpansionError::EmptySequence);
    }

    let len = sequence.len();
    for (index, instruction) in sequence.iter_mut().enumerate() {
        metadata::set_sequence_metadata(instruction, index == 0, Some((len - index - 1) as u16));
    }
    if let Some(last) = sequence.last_mut() {
        last.is_compressed = is_compressed;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finalizes_sequence_metadata() -> Result<(), ExpansionError> {
        let mut allocator = ExpansionAllocator::new();
        let mut assembler = InstrAssembler::new(0x8000_0000, true, &mut allocator);
        assembler.emit_i(InstructionKind::ADDI, 1, 2, 3)?;
        assembler.emit_r(InstructionKind::ADD, 4, 5, 6)?;

        let sequence = assembler.finalize()?;
        assert_eq!(sequence[0].virtual_sequence_remaining, Some(1));
        assert!(sequence[0].is_first_in_sequence);
        assert!(!sequence[0].is_compressed);
        assert_eq!(sequence[1].virtual_sequence_remaining, Some(0));
        assert!(!sequence[1].is_first_in_sequence);
        assert!(sequence[1].is_compressed);
        Ok(())
    }

    #[test]
    fn rejects_inline_write_to_non_inline_register() {
        let mut allocator = ExpansionAllocator::new();
        let mut assembler = InstrAssembler::new_inline(0x8000_0000, false, &mut allocator);

        assert!(matches!(
            assembler.emit_i(InstructionKind::ADDI, 1, 0, 0),
            Err(ExpansionError::InvalidInlineWriteTarget {
                register: 1,
                minimum_register: 40
            })
        ));
    }

    #[test]
    fn finalizes_inline_with_reset_instructions() -> Result<(), ExpansionError> {
        let mut allocator = ExpansionAllocator::new();
        let register = allocator.allocate_for_inline()?;
        allocator.release(register)?;

        let mut assembler = InstrAssembler::new_inline(0x8000_0000, false, &mut allocator);
        assembler.emit_i(InstructionKind::ADDI, register, 0, 7)?;
        let sequence = assembler.finalize_inline()?;

        assert_eq!(sequence.len(), 2);
        assert_eq!(sequence[0].operands.rd, Some(register));
        assert_eq!(sequence[0].operands.imm, 7);
        assert_eq!(sequence[1].instruction_kind, InstructionKind::ADDI);
        assert_eq!(sequence[1].operands.rd, Some(register));
        assert_eq!(sequence[1].operands.imm, 0);
        Ok(())
    }
}
