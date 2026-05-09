use jolt_riscv::{JoltInstructionKind, NormalizedInstruction, NormalizedOperands};

use crate::expand::{
    allocator::ExpansionAllocator,
    core::{ExpansionSequence, ExpansionState},
    ExpansionError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RowTemplate {
    pub(super) instruction_kind: JoltInstructionKind,
    pub(super) operands: NormalizedOperands,
}

impl RowTemplate {
    pub(super) const fn r(instruction_kind: JoltInstructionKind, rd: u8, rs1: u8, rs2: u8) -> Self {
        Self {
            instruction_kind,
            operands: NormalizedOperands {
                rd: Some(rd),
                rs1: Some(rs1),
                rs2: Some(rs2),
                imm: 0,
            },
        }
    }

    pub(super) const fn i(
        instruction_kind: JoltInstructionKind,
        rd: u8,
        rs1: u8,
        imm: i128,
    ) -> Self {
        Self {
            instruction_kind,
            operands: NormalizedOperands {
                rd: Some(rd),
                rs1: Some(rs1),
                rs2: None,
                imm,
            },
        }
    }

    pub(super) const fn j(instruction_kind: JoltInstructionKind, rd: u8, imm: i128) -> Self {
        Self {
            instruction_kind,
            operands: NormalizedOperands {
                rd: Some(rd),
                rs1: None,
                rs2: None,
                imm,
            },
        }
    }

    pub(super) const fn u(instruction_kind: JoltInstructionKind, rd: u8, imm: i128) -> Self {
        Self {
            instruction_kind,
            operands: NormalizedOperands {
                rd: Some(rd),
                rs1: None,
                rs2: None,
                imm,
            },
        }
    }

    pub(super) const fn b(
        instruction_kind: JoltInstructionKind,
        rs1: u8,
        rs2: u8,
        imm: i128,
    ) -> Self {
        Self {
            instruction_kind,
            operands: NormalizedOperands {
                rd: None,
                rs1: Some(rs1),
                rs2: Some(rs2),
                imm,
            },
        }
    }

    pub(super) const fn s(
        instruction_kind: JoltInstructionKind,
        rs1: u8,
        rs2: u8,
        imm: i128,
    ) -> Self {
        Self {
            instruction_kind,
            operands: NormalizedOperands {
                rd: None,
                rs1: Some(rs1),
                rs2: Some(rs2),
                imm,
            },
        }
    }

    /// Pseudo-I format for address/alignment assertions that read `rs1` and an
    /// immediate offset but do not write `rd`.
    pub(super) const fn address(instruction_kind: JoltInstructionKind, rs1: u8, imm: i128) -> Self {
        Self {
            instruction_kind,
            operands: NormalizedOperands {
                rd: None,
                rs1: Some(rs1),
                rs2: None,
                imm,
            },
        }
    }

    pub(super) fn instruction_at(self, address: usize) -> NormalizedInstruction {
        NormalizedInstruction {
            instruction_kind: self.instruction_kind,
            address,
            operands: self.operands,
            virtual_sequence_remaining: Some(0),
            is_first_in_sequence: false,
            is_compressed: false,
        }
    }
}

pub(super) struct ExpansionBuilder<'a, 'b> {
    source: &'b NormalizedInstruction,
    state: ExpansionState<'a>,
    sequence: ExpansionSequence,
}

impl<'a, 'b> ExpansionBuilder<'a, 'b> {
    pub(super) fn new(
        source: &'b NormalizedInstruction,
        allocator: &'a mut ExpansionAllocator,
    ) -> Self {
        Self {
            source,
            state: ExpansionState::new(allocator),
            sequence: ExpansionSequence::new(source),
        }
    }

    pub(super) fn allocate(&mut self) -> Result<u8, ExpansionError> {
        self.state.allocator().allocate()
    }

    /// Append an already target-legal row to this source row's output sequence.
    ///
    /// Use `emit_*` when the row should appear exactly as written in finalized
    /// bytecode. Use `expand_*` instead when the row is a source-only helper
    /// that must be routed through the central expander first.
    pub(super) fn emit_r(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rd: u8,
        rs1: u8,
        rs2: u8,
    ) {
        self.emit(RowTemplate::r(instruction_kind, rd, rs1, rs2));
    }

    pub(super) fn emit_i(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rd: u8,
        rs1: u8,
        imm: i128,
    ) {
        self.emit(RowTemplate::i(instruction_kind, rd, rs1, imm));
    }

    pub(super) fn emit_j(&mut self, instruction_kind: JoltInstructionKind, rd: u8, imm: i128) {
        self.emit(RowTemplate::j(instruction_kind, rd, imm));
    }

    pub(super) fn emit_u(&mut self, instruction_kind: JoltInstructionKind, rd: u8, imm: i128) {
        self.emit(RowTemplate::u(instruction_kind, rd, imm));
    }

    /// Route a source-only helper row through provider-free expansion, then
    /// append the resulting finalized rows to this source row's output sequence.
    ///
    /// Recursive helper expansion always goes through `ExpansionState`, so
    /// rd=x0 handling, recursion depth, allocator state, and metadata stamping
    /// stay centralized.
    pub(super) fn expand_r(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rd: u8,
        rs1: u8,
        rs2: u8,
    ) -> Result<(), ExpansionError> {
        self.expand(RowTemplate::r(instruction_kind, rd, rs1, rs2))
    }

    pub(super) fn expand_i(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rd: u8,
        rs1: u8,
        imm: i128,
    ) -> Result<(), ExpansionError> {
        self.expand(RowTemplate::i(instruction_kind, rd, rs1, imm))
    }

    pub(super) fn expand_j(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rd: u8,
        imm: i128,
    ) -> Result<(), ExpansionError> {
        self.expand(RowTemplate::j(instruction_kind, rd, imm))
    }

    pub(super) fn expand_u(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rd: u8,
        imm: i128,
    ) -> Result<(), ExpansionError> {
        self.expand(RowTemplate::u(instruction_kind, rd, imm))
    }

    pub(super) fn expand_b(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rs1: u8,
        rs2: u8,
        imm: i128,
    ) -> Result<(), ExpansionError> {
        self.expand(RowTemplate::b(instruction_kind, rs1, rs2, imm))
    }

    pub(super) fn expand_s(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rs1: u8,
        rs2: u8,
        imm: i128,
    ) -> Result<(), ExpansionError> {
        self.expand(RowTemplate::s(instruction_kind, rs1, rs2, imm))
    }

    pub(super) fn expand_address(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rs1: u8,
        imm: i128,
    ) -> Result<(), ExpansionError> {
        self.expand(RowTemplate::address(instruction_kind, rs1, imm))
    }

    pub(super) fn release(&mut self, register: u8) -> Result<(), ExpansionError> {
        self.state.allocator().release(register)
    }

    pub(super) fn release_many(
        &mut self,
        registers: impl IntoIterator<Item = u8>,
    ) -> Result<(), ExpansionError> {
        for register in registers {
            self.release(register)?;
        }
        Ok(())
    }

    pub(super) fn finalize(self) -> Result<Vec<NormalizedInstruction>, ExpansionError> {
        self.sequence.finish()
    }

    fn emit(&mut self, row: RowTemplate) {
        self.sequence.emit(row.instruction_kind, row.operands);
    }

    fn expand(&mut self, row: RowTemplate) -> Result<(), ExpansionError> {
        let instruction = row.instruction_at(self.source.address);
        self.sequence
            .extend(self.state.expand_one_core(&instruction)?)
    }
}

pub(super) fn is_source_only(instruction_kind: JoltInstructionKind) -> bool {
    matches!(
        instruction_kind,
        JoltInstructionKind::Inline
            | JoltInstructionKind::ADDIW
            | JoltInstructionKind::ADDW
            | JoltInstructionKind::SUBW
            | JoltInstructionKind::MULH
            | JoltInstructionKind::MULHSU
            | JoltInstructionKind::MULW
            | JoltInstructionKind::LB
            | JoltInstructionKind::LBU
            | JoltInstructionKind::LH
            | JoltInstructionKind::LHU
            | JoltInstructionKind::LW
            | JoltInstructionKind::LWU
            | JoltInstructionKind::AdviceLB
            | JoltInstructionKind::AdviceLH
            | JoltInstructionKind::AdviceLW
            | JoltInstructionKind::AdviceLD
            | JoltInstructionKind::AMOADDD
            | JoltInstructionKind::AMOANDD
            | JoltInstructionKind::AMOORD
            | JoltInstructionKind::AMOXORD
            | JoltInstructionKind::AMOSWAPD
            | JoltInstructionKind::AMOMAXD
            | JoltInstructionKind::AMOMAXUD
            | JoltInstructionKind::AMOMIND
            | JoltInstructionKind::AMOMINUD
            | JoltInstructionKind::AMOADDW
            | JoltInstructionKind::AMOANDW
            | JoltInstructionKind::AMOORW
            | JoltInstructionKind::AMOXORW
            | JoltInstructionKind::AMOSWAPW
            | JoltInstructionKind::AMOMAXW
            | JoltInstructionKind::AMOMAXUW
            | JoltInstructionKind::AMOMINW
            | JoltInstructionKind::AMOMINUW
            | JoltInstructionKind::LRD
            | JoltInstructionKind::LRW
            | JoltInstructionKind::DIV
            | JoltInstructionKind::DIVU
            | JoltInstructionKind::DIVW
            | JoltInstructionKind::DIVUW
            | JoltInstructionKind::REM
            | JoltInstructionKind::REMU
            | JoltInstructionKind::REMW
            | JoltInstructionKind::REMUW
            | JoltInstructionKind::SB
            | JoltInstructionKind::SCD
            | JoltInstructionKind::SCW
            | JoltInstructionKind::SH
            | JoltInstructionKind::SW
            | JoltInstructionKind::CSRRW
            | JoltInstructionKind::CSRRS
            | JoltInstructionKind::EBREAK
            | JoltInstructionKind::ECALL
            | JoltInstructionKind::MRET
            | JoltInstructionKind::SLL
            | JoltInstructionKind::SLLI
            | JoltInstructionKind::SLLW
            | JoltInstructionKind::SLLIW
            | JoltInstructionKind::SRL
            | JoltInstructionKind::SRLI
            | JoltInstructionKind::SRA
            | JoltInstructionKind::SRAI
            | JoltInstructionKind::SRLIW
            | JoltInstructionKind::SRAIW
            | JoltInstructionKind::SRLW
            | JoltInstructionKind::SRAW
    )
}

pub(super) fn is_target_legal(instruction_kind: JoltInstructionKind) -> bool {
    !is_source_only(instruction_kind)
}
