use jolt_riscv::{JoltInstructionKind, NormalizedInstruction, NormalizedOperands};

use crate::expand::{allocator::ExpansionAllocator, expand_instruction, ExpansionError};

pub(super) struct ExpansionSequence {
    address: usize,
    is_compressed: bool,
    rows: Vec<NormalizedInstruction>,
}

impl ExpansionSequence {
    pub(super) fn new(source: &NormalizedInstruction) -> Self {
        Self {
            address: source.address,
            is_compressed: source.is_compressed,
            rows: Vec::new(),
        }
    }

    pub(super) fn emit(
        &mut self,
        instruction_kind: JoltInstructionKind,
        operands: NormalizedOperands,
    ) {
        self.rows.push(NormalizedInstruction {
            instruction_kind,
            address: self.address,
            operands,
            virtual_sequence_remaining: Some(0),
            is_first_in_sequence: false,
            is_compressed: false,
        });
    }

    pub(super) fn emit_r(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rd: u8,
        rs1: u8,
        rs2: u8,
    ) {
        self.emit(
            instruction_kind,
            NormalizedOperands {
                rd: Some(rd),
                rs1: Some(rs1),
                rs2: Some(rs2),
                imm: 0,
            },
        );
    }

    pub(super) fn emit_i(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rd: u8,
        rs1: u8,
        imm: i128,
    ) {
        self.emit(
            instruction_kind,
            NormalizedOperands {
                rd: Some(rd),
                rs1: Some(rs1),
                rs2: None,
                imm,
            },
        );
    }

    pub(super) fn emit_j(&mut self, instruction_kind: JoltInstructionKind, rd: u8, imm: i128) {
        self.emit(
            instruction_kind,
            NormalizedOperands {
                rd: Some(rd),
                rs1: None,
                rs2: None,
                imm,
            },
        );
    }

    pub(super) fn emit_u(&mut self, instruction_kind: JoltInstructionKind, rd: u8, imm: i128) {
        self.emit(
            instruction_kind,
            NormalizedOperands {
                rd: Some(rd),
                rs1: None,
                rs2: None,
                imm,
            },
        );
    }

    pub(super) fn emit_expanded(
        &mut self,
        instruction_kind: JoltInstructionKind,
        operands: NormalizedOperands,
        allocator: &mut ExpansionAllocator,
    ) -> Result<(), ExpansionError> {
        let instruction = NormalizedInstruction {
            instruction_kind,
            address: self.address,
            operands,
            virtual_sequence_remaining: Some(0),
            is_first_in_sequence: false,
            is_compressed: false,
        };
        self.rows
            .extend(expand_instruction(&instruction, allocator)?);
        Ok(())
    }

    pub(super) fn emit_r_expanded(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rd: u8,
        rs1: u8,
        rs2: u8,
        allocator: &mut ExpansionAllocator,
    ) -> Result<(), ExpansionError> {
        self.emit_expanded(
            instruction_kind,
            NormalizedOperands {
                rd: Some(rd),
                rs1: Some(rs1),
                rs2: Some(rs2),
                imm: 0,
            },
            allocator,
        )
    }

    pub(super) fn emit_i_expanded(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rd: u8,
        rs1: u8,
        imm: i128,
        allocator: &mut ExpansionAllocator,
    ) -> Result<(), ExpansionError> {
        self.emit_expanded(
            instruction_kind,
            NormalizedOperands {
                rd: Some(rd),
                rs1: Some(rs1),
                rs2: None,
                imm,
            },
            allocator,
        )
    }

    pub(super) fn emit_s_expanded(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rs1: u8,
        rs2: u8,
        imm: i128,
        allocator: &mut ExpansionAllocator,
    ) -> Result<(), ExpansionError> {
        self.emit_expanded(
            instruction_kind,
            NormalizedOperands {
                rd: None,
                rs1: Some(rs1),
                rs2: Some(rs2),
                imm,
            },
            allocator,
        )
    }

    pub(super) fn emit_b_expanded(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rs1: u8,
        rs2: u8,
        imm: i128,
        allocator: &mut ExpansionAllocator,
    ) -> Result<(), ExpansionError> {
        self.emit_expanded(
            instruction_kind,
            NormalizedOperands {
                rd: None,
                rs1: Some(rs1),
                rs2: Some(rs2),
                imm,
            },
            allocator,
        )
    }

    pub(super) fn emit_j_expanded(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rd: u8,
        imm: i128,
        allocator: &mut ExpansionAllocator,
    ) -> Result<(), ExpansionError> {
        self.emit_expanded(
            instruction_kind,
            NormalizedOperands {
                rd: Some(rd),
                rs1: None,
                rs2: None,
                imm,
            },
            allocator,
        )
    }

    pub(super) fn emit_u_expanded(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rd: u8,
        imm: i128,
        allocator: &mut ExpansionAllocator,
    ) -> Result<(), ExpansionError> {
        self.emit_expanded(
            instruction_kind,
            NormalizedOperands {
                rd: Some(rd),
                rs1: None,
                rs2: None,
                imm,
            },
            allocator,
        )
    }

    pub(super) fn emit_align_expanded(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rs1: u8,
        imm: i128,
        allocator: &mut ExpansionAllocator,
    ) -> Result<(), ExpansionError> {
        self.emit_expanded(
            instruction_kind,
            NormalizedOperands {
                rd: None,
                rs1: Some(rs1),
                rs2: None,
                imm,
            },
            allocator,
        )
    }

    pub(super) fn finish(mut self) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
        if self.rows.is_empty() {
            return Err(ExpansionError::EmptySequence);
        }

        let len = self.rows.len();
        for (index, row) in self.rows.iter_mut().enumerate() {
            row.is_first_in_sequence = index == 0;
            row.virtual_sequence_remaining = Some((len - index - 1) as u16);
        }
        if let Some(last) = self.rows.last_mut() {
            last.is_compressed = self.is_compressed;
        }
        Ok(self.rows)
    }

    pub(super) fn finish_releasing(
        self,
        allocator: &mut ExpansionAllocator,
        registers: impl IntoIterator<Item = u8>,
    ) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
        let rows = self.finish()?;
        for register in registers {
            allocator.release(register)?;
        }
        Ok(rows)
    }
}
