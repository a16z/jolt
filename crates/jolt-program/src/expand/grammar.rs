use jolt_riscv::{JoltInstructionKind, JoltRow};

use crate::expand::{allocator::NUM_VIRTUAL_INSTRUCTION_REGISTERS, ExpansionError};

/// Symbolic register placeholder, resolved to a physical virtual register during materialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TempId(pub(super) u8);

impl TempId {
    pub(super) const fn index(self) -> usize {
        self.0 as usize
    }

    pub(super) const fn operand(self) -> RegisterOperand {
        RegisterOperand::Temp(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RegisterOperand {
    Register(u8),
    Temp(TempId),
}

pub(super) const fn reg(register: u8) -> RegisterOperand {
    RegisterOperand::Register(register)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TemplateOperands {
    pub(super) rd: Option<RegisterOperand>,
    pub(super) rs1: Option<RegisterOperand>,
    pub(super) rs2: Option<RegisterOperand>,
    pub(super) imm: i128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RowTemplate {
    pub(super) instruction_kind: JoltInstructionKind,
    pub(super) operands: TemplateOperands,
}

impl RowTemplate {
    pub(super) fn r(
        instruction_kind: JoltInstructionKind,
        rd: RegisterOperand,
        rs1: RegisterOperand,
        rs2: RegisterOperand,
    ) -> Self {
        Self {
            instruction_kind,
            operands: TemplateOperands {
                rd: Some(rd),
                rs1: Some(rs1),
                rs2: Some(rs2),
                imm: 0,
            },
        }
    }

    pub(super) fn i(
        instruction_kind: JoltInstructionKind,
        rd: RegisterOperand,
        rs1: RegisterOperand,
        imm: i128,
    ) -> Self {
        Self {
            instruction_kind,
            operands: TemplateOperands {
                rd: Some(rd),
                rs1: Some(rs1),
                rs2: None,
                imm,
            },
        }
    }

    pub(super) fn j(instruction_kind: JoltInstructionKind, rd: RegisterOperand, imm: i128) -> Self {
        Self {
            instruction_kind,
            operands: TemplateOperands {
                rd: Some(rd),
                rs1: None,
                rs2: None,
                imm,
            },
        }
    }

    pub(super) fn u(instruction_kind: JoltInstructionKind, rd: RegisterOperand, imm: i128) -> Self {
        Self {
            instruction_kind,
            operands: TemplateOperands {
                rd: Some(rd),
                rs1: None,
                rs2: None,
                imm,
            },
        }
    }

    pub(super) fn b(
        instruction_kind: JoltInstructionKind,
        rs1: RegisterOperand,
        rs2: RegisterOperand,
        imm: i128,
    ) -> Self {
        Self {
            instruction_kind,
            operands: TemplateOperands {
                rd: None,
                rs1: Some(rs1),
                rs2: Some(rs2),
                imm,
            },
        }
    }

    pub(super) fn s(
        instruction_kind: JoltInstructionKind,
        rs1: RegisterOperand,
        rs2: RegisterOperand,
        imm: i128,
    ) -> Self {
        Self {
            instruction_kind,
            operands: TemplateOperands {
                rd: None,
                rs1: Some(rs1),
                rs2: Some(rs2),
                imm,
            },
        }
    }

    /// Pseudo-I format for address/alignment assertions that read `rs1` and an
    /// immediate offset but do not write `rd`.
    pub(super) fn address(
        instruction_kind: JoltInstructionKind,
        rs1: RegisterOperand,
        imm: i128,
    ) -> Self {
        Self {
            instruction_kind,
            operands: TemplateOperands {
                rd: None,
                rs1: Some(rs1),
                rs2: None,
                imm,
            },
        }
    }
}

/// A single step in a symbolic expansion recipe.
#[derive(Clone, Copy)]
pub(super) enum ExpansionOp {
    /// Append this row directly to the output.
    Emit(RowTemplate),
    /// Recursively expand this row through the full pipeline before appending.
    Expand(RowTemplate),
    Allocate(TempId),
    Release(TempId),
}

/// A complete symbolic recipe: source instruction paired with the ops to materialize it.
pub(super) struct ExpandedInstructionSequence {
    pub(super) source: JoltRow,
    pub(super) ops: Vec<ExpansionOp>,
}

/// Builds a symbolic expansion recipe from emit/expand/allocate/release calls.
pub(super) struct ExpansionBuilder {
    source: JoltRow,
    ops: Vec<ExpansionOp>,
    next_temp: usize,
}

impl ExpansionBuilder {
    pub(super) fn new(source: JoltRow) -> Self {
        Self {
            source,
            ops: Vec::new(),
            next_temp: 0,
        }
    }

    pub(super) fn allocate(&mut self) -> Result<TempId, ExpansionError> {
        if self.next_temp >= NUM_VIRTUAL_INSTRUCTION_REGISTERS {
            return Err(ExpansionError::TooManyTemporaryRegisters {
                actual: self.next_temp + 1,
            });
        }
        let temp = TempId(self.next_temp as u8);
        self.next_temp += 1;
        self.ops.push(ExpansionOp::Allocate(temp));
        Ok(temp)
    }

    /// Append an already target-legal row to this source row's output sequence.
    ///
    /// Use `emit_*` when the row should appear exactly as written in finalized
    /// bytecode. Use `expand_*` instead when the row is a source-only helper
    /// that must be routed through the central expander first.
    pub(super) fn emit_r(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rd: RegisterOperand,
        rs1: RegisterOperand,
        rs2: RegisterOperand,
    ) {
        self.emit(RowTemplate::r(instruction_kind, rd, rs1, rs2));
    }

    pub(super) fn emit_i(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rd: RegisterOperand,
        rs1: RegisterOperand,
        imm: i128,
    ) {
        self.emit(RowTemplate::i(instruction_kind, rd, rs1, imm));
    }

    pub(super) fn emit_j(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rd: RegisterOperand,
        imm: i128,
    ) {
        self.emit(RowTemplate::j(instruction_kind, rd, imm));
    }

    pub(super) fn emit_u(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rd: RegisterOperand,
        imm: i128,
    ) {
        self.emit(RowTemplate::u(instruction_kind, rd, imm));
    }

    /// Record a source-only helper row that the provider-free materializer must
    /// expand before appending its finalized rows to this source-row sequence.
    ///
    /// Recursive helper expansion always goes through `ExpansionState`, so
    /// rd=x0 handling, recursion depth, allocator state, and metadata stamping
    /// stay centralized.
    pub(super) fn expand_r(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rd: RegisterOperand,
        rs1: RegisterOperand,
        rs2: RegisterOperand,
    ) {
        self.expand(RowTemplate::r(instruction_kind, rd, rs1, rs2));
    }

    pub(super) fn expand_i(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rd: RegisterOperand,
        rs1: RegisterOperand,
        imm: i128,
    ) {
        self.expand(RowTemplate::i(instruction_kind, rd, rs1, imm));
    }

    pub(super) fn expand_j(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rd: RegisterOperand,
        imm: i128,
    ) {
        self.expand(RowTemplate::j(instruction_kind, rd, imm));
    }

    pub(super) fn expand_u(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rd: RegisterOperand,
        imm: i128,
    ) {
        self.expand(RowTemplate::u(instruction_kind, rd, imm));
    }

    pub(super) fn expand_b(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rs1: RegisterOperand,
        rs2: RegisterOperand,
        imm: i128,
    ) {
        self.expand(RowTemplate::b(instruction_kind, rs1, rs2, imm));
    }

    pub(super) fn expand_s(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rs1: RegisterOperand,
        rs2: RegisterOperand,
        imm: i128,
    ) {
        self.expand(RowTemplate::s(instruction_kind, rs1, rs2, imm));
    }

    pub(super) fn expand_address(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rs1: RegisterOperand,
        imm: i128,
    ) {
        self.expand(RowTemplate::address(instruction_kind, rs1, imm));
    }

    pub(super) fn release(&mut self, temp: TempId) {
        self.ops.push(ExpansionOp::Release(temp));
    }

    pub(super) fn release_many<const N: usize>(&mut self, registers: [TempId; N]) {
        for register in registers {
            self.release(register);
        }
    }

    pub(super) fn finalize(self) -> Result<ExpandedInstructionSequence, ExpansionError> {
        Ok(ExpandedInstructionSequence {
            source: self.source,
            ops: self.ops,
        })
    }

    fn emit(&mut self, row: RowTemplate) {
        self.ops.push(ExpansionOp::Emit(row));
    }

    fn expand(&mut self, row: RowTemplate) {
        self.ops.push(ExpansionOp::Expand(row));
    }
}

/// Instructions that exist only in decoded source and must be expanded into target-legal sequences.
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

#[cfg(test)]
mod tests {
    use jolt_riscv::{JoltInstructionKind, JoltRow, NormalizedOperands};

    use super::*;

    fn source() -> JoltRow {
        JoltRow {
            instruction_kind: JoltInstructionKind::ADDIW,
            address: 0x8000_0000,
            operands: NormalizedOperands::default(),
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        }
    }

    #[test]
    fn symbolic_temps_are_limited_to_instruction_register_pool() -> Result<(), ExpansionError> {
        let mut builder = ExpansionBuilder::new(source());
        for _ in 0..NUM_VIRTUAL_INSTRUCTION_REGISTERS {
            let _ = builder.allocate()?;
        }

        assert!(matches!(
            builder.allocate(),
            Err(ExpansionError::TooManyTemporaryRegisters { actual })
                if actual == NUM_VIRTUAL_INSTRUCTION_REGISTERS + 1
        ));
        Ok(())
    }
}
