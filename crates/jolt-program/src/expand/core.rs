use jolt_riscv::{JoltInstructionKind, NormalizedInstruction, NormalizedOperands};

use crate::expand::{
    allocator::ExpansionAllocator, buffer::ExpansionBuffer, expand_instruction_core,
    grammar::ExpansionOp, metadata::stamp_sequence, ExpansionError,
};

pub(super) struct ExpansionState<'a> {
    allocator: &'a mut ExpansionAllocator,
}

impl<'a> ExpansionState<'a> {
    pub(super) fn new(allocator: &'a mut ExpansionAllocator) -> Self {
        Self { allocator }
    }

    pub(super) fn allocator(&mut self) -> &mut ExpansionAllocator {
        self.allocator
    }

    pub(super) fn expand_one_core(
        &mut self,
        instruction: &NormalizedInstruction,
    ) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
        self.allocator.enter_expansion()?;
        let result = expand_instruction_core(instruction, self);
        self.allocator.exit_expansion();
        result
    }
}

pub(super) struct ExpansionSequence {
    address: usize,
    is_compressed: bool,
    rows: ExpansionBuffer,
}

impl ExpansionSequence {
    pub(super) fn new(source: &NormalizedInstruction) -> Self {
        Self {
            address: source.address,
            is_compressed: source.is_compressed,
            rows: ExpansionBuffer::new(),
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

    pub(super) fn emit_op(&mut self, op: ExpansionOp) {
        match op {
            ExpansionOp::Row(row) => self.emit(row.instruction_kind, row.operands),
        }
    }

    pub(super) fn emit_ops(&mut self, ops: impl IntoIterator<Item = ExpansionOp>) {
        for op in ops {
            self.emit_op(op);
        }
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
            .extend(ExpansionState::new(allocator).expand_one_core(&instruction)?)
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

    pub(super) fn finish(self) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
        self.rows.check_capacity()?;
        stamp_sequence(self.rows.into_vec(), self.is_compressed)
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
