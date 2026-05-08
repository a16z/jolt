use jolt_riscv::{JoltInstructionKind, NormalizedInstruction, NormalizedOperands};

use crate::expand::{allocator::ExpansionAllocator, ExpansionError};

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
