use jolt_riscv::{JoltInstructionKind, JoltRow, SourceInstructionKind};

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
pub(super) struct InstructionTemplate<K> {
    pub(super) instruction_kind: K,
    pub(super) operands: TemplateOperands,
}

pub(super) type RowTemplate = InstructionTemplate<JoltInstructionKind>;
pub(super) type SourceRowTemplate = InstructionTemplate<SourceInstructionKind>;

impl<K> InstructionTemplate<K> {
    pub(super) fn r(
        instruction_kind: K,
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
        instruction_kind: K,
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

    pub(super) fn j(instruction_kind: K, rd: RegisterOperand, imm: i128) -> Self {
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

    pub(super) fn u(instruction_kind: K, rd: RegisterOperand, imm: i128) -> Self {
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
        instruction_kind: K,
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
        instruction_kind: K,
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
    pub(super) fn address(instruction_kind: K, rs1: RegisterOperand, imm: i128) -> Self {
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
    Expand(SourceRowTemplate),
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
        instruction_kind: SourceInstructionKind,
        rd: RegisterOperand,
        rs1: RegisterOperand,
        rs2: RegisterOperand,
    ) {
        self.expand(SourceRowTemplate::r(instruction_kind, rd, rs1, rs2));
    }

    pub(super) fn expand_i(
        &mut self,
        instruction_kind: SourceInstructionKind,
        rd: RegisterOperand,
        rs1: RegisterOperand,
        imm: i128,
    ) {
        self.expand(SourceRowTemplate::i(instruction_kind, rd, rs1, imm));
    }

    pub(super) fn expand_j(
        &mut self,
        instruction_kind: SourceInstructionKind,
        rd: RegisterOperand,
        imm: i128,
    ) {
        self.expand(SourceRowTemplate::j(instruction_kind, rd, imm));
    }

    pub(super) fn expand_u(
        &mut self,
        instruction_kind: SourceInstructionKind,
        rd: RegisterOperand,
        imm: i128,
    ) {
        self.expand(SourceRowTemplate::u(instruction_kind, rd, imm));
    }

    pub(super) fn expand_b(
        &mut self,
        instruction_kind: SourceInstructionKind,
        rs1: RegisterOperand,
        rs2: RegisterOperand,
        imm: i128,
    ) {
        self.expand(SourceRowTemplate::b(instruction_kind, rs1, rs2, imm));
    }

    pub(super) fn expand_s(
        &mut self,
        instruction_kind: SourceInstructionKind,
        rs1: RegisterOperand,
        rs2: RegisterOperand,
        imm: i128,
    ) {
        self.expand(SourceRowTemplate::s(instruction_kind, rs1, rs2, imm));
    }

    pub(super) fn expand_address(
        &mut self,
        instruction_kind: SourceInstructionKind,
        rs1: RegisterOperand,
        imm: i128,
    ) {
        self.expand(SourceRowTemplate::address(instruction_kind, rs1, imm));
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

    fn expand(&mut self, row: SourceRowTemplate) {
        self.ops.push(ExpansionOp::Expand(row));
    }
}

/// Instructions that exist only in decoded source and must be expanded into target-legal sequences.
pub(super) fn is_source_only(instruction_kind: SourceInstructionKind) -> bool {
    matches!(
        instruction_kind,
        SourceInstructionKind::Inline
            | SourceInstructionKind::ADDIW
            | SourceInstructionKind::ADDW
            | SourceInstructionKind::SUBW
            | SourceInstructionKind::MULH
            | SourceInstructionKind::MULHSU
            | SourceInstructionKind::MULW
            | SourceInstructionKind::LB
            | SourceInstructionKind::LBU
            | SourceInstructionKind::LH
            | SourceInstructionKind::LHU
            | SourceInstructionKind::LW
            | SourceInstructionKind::LWU
            | SourceInstructionKind::AdviceLB
            | SourceInstructionKind::AdviceLH
            | SourceInstructionKind::AdviceLW
            | SourceInstructionKind::AdviceLD
            | SourceInstructionKind::AMOADDD
            | SourceInstructionKind::AMOANDD
            | SourceInstructionKind::AMOORD
            | SourceInstructionKind::AMOXORD
            | SourceInstructionKind::AMOSWAPD
            | SourceInstructionKind::AMOMAXD
            | SourceInstructionKind::AMOMAXUD
            | SourceInstructionKind::AMOMIND
            | SourceInstructionKind::AMOMINUD
            | SourceInstructionKind::AMOADDW
            | SourceInstructionKind::AMOANDW
            | SourceInstructionKind::AMOORW
            | SourceInstructionKind::AMOXORW
            | SourceInstructionKind::AMOSWAPW
            | SourceInstructionKind::AMOMAXW
            | SourceInstructionKind::AMOMAXUW
            | SourceInstructionKind::AMOMINW
            | SourceInstructionKind::AMOMINUW
            | SourceInstructionKind::LRD
            | SourceInstructionKind::LRW
            | SourceInstructionKind::DIV
            | SourceInstructionKind::DIVU
            | SourceInstructionKind::DIVW
            | SourceInstructionKind::DIVUW
            | SourceInstructionKind::REM
            | SourceInstructionKind::REMU
            | SourceInstructionKind::REMW
            | SourceInstructionKind::REMUW
            | SourceInstructionKind::SB
            | SourceInstructionKind::SCD
            | SourceInstructionKind::SCW
            | SourceInstructionKind::SH
            | SourceInstructionKind::SW
            | SourceInstructionKind::CSRRW
            | SourceInstructionKind::CSRRS
            | SourceInstructionKind::EBREAK
            | SourceInstructionKind::ECALL
            | SourceInstructionKind::MRET
            | SourceInstructionKind::SLL
            | SourceInstructionKind::SLLI
            | SourceInstructionKind::SLLW
            | SourceInstructionKind::SLLIW
            | SourceInstructionKind::SRL
            | SourceInstructionKind::SRLI
            | SourceInstructionKind::SRA
            | SourceInstructionKind::SRAI
            | SourceInstructionKind::SRLIW
            | SourceInstructionKind::SRAIW
            | SourceInstructionKind::SRLW
            | SourceInstructionKind::SRAW
    )
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
