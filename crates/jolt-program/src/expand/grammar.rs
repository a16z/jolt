use jolt_riscv::{JoltInstructionKind, NormalizedInstruction, SourceInstructionKind};

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
pub(super) struct RowTemplate<K = JoltInstructionKind> {
    pub(super) instruction_kind: K,
    pub(super) operands: TemplateOperands,
}

impl<K> RowTemplate<K> {
    pub(super) fn r(
        instruction_kind: impl Into<K>,
        rd: RegisterOperand,
        rs1: RegisterOperand,
        rs2: RegisterOperand,
    ) -> Self {
        Self {
            instruction_kind: instruction_kind.into(),
            operands: TemplateOperands {
                rd: Some(rd),
                rs1: Some(rs1),
                rs2: Some(rs2),
                imm: 0,
            },
        }
    }

    pub(super) fn i(
        instruction_kind: impl Into<K>,
        rd: RegisterOperand,
        rs1: RegisterOperand,
        imm: i128,
    ) -> Self {
        Self {
            instruction_kind: instruction_kind.into(),
            operands: TemplateOperands {
                rd: Some(rd),
                rs1: Some(rs1),
                rs2: None,
                imm,
            },
        }
    }

    pub(super) fn j(instruction_kind: impl Into<K>, rd: RegisterOperand, imm: i128) -> Self {
        Self {
            instruction_kind: instruction_kind.into(),
            operands: TemplateOperands {
                rd: Some(rd),
                rs1: None,
                rs2: None,
                imm,
            },
        }
    }

    pub(super) fn u(instruction_kind: impl Into<K>, rd: RegisterOperand, imm: i128) -> Self {
        Self {
            instruction_kind: instruction_kind.into(),
            operands: TemplateOperands {
                rd: Some(rd),
                rs1: None,
                rs2: None,
                imm,
            },
        }
    }
    pub(super) fn b(
        instruction_kind: impl Into<K>,
        rs1: RegisterOperand,
        rs2: RegisterOperand,
        imm: i128,
    ) -> Self {
        Self {
            instruction_kind: instruction_kind.into(),
            operands: TemplateOperands {
                rd: None,
                rs1: Some(rs1),
                rs2: Some(rs2),
                imm,
            },
        }
    }

    pub(super) fn s(
        instruction_kind: impl Into<K>,
        rs1: RegisterOperand,
        rs2: RegisterOperand,
        imm: i128,
    ) -> Self {
        Self {
            instruction_kind: instruction_kind.into(),
            operands: TemplateOperands {
                rd: None,
                rs1: Some(rs1),
                rs2: Some(rs2),
                imm,
            },
        }
    }

    pub(super) fn address(instruction_kind: impl Into<K>, rs1: RegisterOperand, imm: i128) -> Self {
        Self {
            instruction_kind: instruction_kind.into(),
            operands: TemplateOperands {
                rd: None,
                rs1: Some(rs1),
                rs2: None,
                imm,
            },
        }
    }
}

/// Kind carried by a recursive dispatch recipe step.
///
/// `Final` rows are already expressed in final Jolt bytecode vocabulary.
/// `Source` rows intentionally request another source/helper lowering pass.
/// The enum stays private to expansion so the public API remains the simpler
/// `SourceInstruction -> Vec<NormalizedInstruction>` boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DispatchInstructionKind {
    Source(SourceInstructionKind),
    Final(JoltInstructionKind),
}

impl From<SourceInstructionKind> for DispatchInstructionKind {
    fn from(value: SourceInstructionKind) -> Self {
        Self::Source(value)
    }
}

impl From<JoltInstructionKind> for DispatchInstructionKind {
    fn from(value: JoltInstructionKind) -> Self {
        Self::Final(value)
    }
}

pub(super) type DispatchRowTemplate = RowTemplate<DispatchInstructionKind>;

/// A single step in a symbolic expansion recipe.
#[derive(Clone, Copy)]
pub(super) enum ExpansionOp {
    /// Append this row directly to the output.
    Emit(RowTemplate),
    /// Recursively expand this row through the full pipeline before appending.
    Dispatch(DispatchRowTemplate),
    Allocate(TempId),
    Release(TempId),
}

/// A complete symbolic recipe: source instruction paired with the ops to materialize it.
pub(super) struct ExpandedInstructionSequence {
    pub(super) source: NormalizedInstruction,
    pub(super) ops: Vec<ExpansionOp>,
}

/// Builds a symbolic expansion recipe from emit/dispatch/allocate/release calls.
pub(super) struct ExpansionBuilder {
    source: NormalizedInstruction,
    ops: Vec<ExpansionOp>,
    next_temp: usize,
}

impl ExpansionBuilder {
    pub(super) fn new(source: NormalizedInstruction) -> Self {
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
    /// bytecode. Use `dispatch_*` when the row must first go through recursive
    /// canonicalization, including rd=x0 handling and source-only lowering.
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

    /// Record a row that the provider-free materializer must dispatch before
    /// appending its finalized rows to this source-row sequence.
    ///
    /// Recursive helper dispatch always goes through `ExpansionState`, so
    /// rd=x0 handling, source-only lowering, recursion depth, allocator state,
    /// and metadata stamping stay centralized.
    pub(super) fn dispatch_r(
        &mut self,
        instruction_kind: impl Into<DispatchInstructionKind>,
        rd: RegisterOperand,
        rs1: RegisterOperand,
        rs2: RegisterOperand,
    ) {
        self.dispatch(DispatchRowTemplate::r(instruction_kind, rd, rs1, rs2));
    }

    pub(super) fn dispatch_i(
        &mut self,
        instruction_kind: impl Into<DispatchInstructionKind>,
        rd: RegisterOperand,
        rs1: RegisterOperand,
        imm: i128,
    ) {
        self.dispatch(DispatchRowTemplate::i(instruction_kind, rd, rs1, imm));
    }

    pub(super) fn dispatch_j(
        &mut self,
        instruction_kind: impl Into<DispatchInstructionKind>,
        rd: RegisterOperand,
        imm: i128,
    ) {
        self.dispatch(DispatchRowTemplate::j(instruction_kind, rd, imm));
    }

    pub(super) fn dispatch_u(
        &mut self,
        instruction_kind: impl Into<DispatchInstructionKind>,
        rd: RegisterOperand,
        imm: i128,
    ) {
        self.dispatch(DispatchRowTemplate::u(instruction_kind, rd, imm));
    }

    pub(super) fn dispatch_b(
        &mut self,
        instruction_kind: impl Into<DispatchInstructionKind>,
        rs1: RegisterOperand,
        rs2: RegisterOperand,
        imm: i128,
    ) {
        self.dispatch(DispatchRowTemplate::b(instruction_kind, rs1, rs2, imm));
    }

    pub(super) fn dispatch_s(
        &mut self,
        instruction_kind: impl Into<DispatchInstructionKind>,
        rs1: RegisterOperand,
        rs2: RegisterOperand,
        imm: i128,
    ) {
        self.dispatch(DispatchRowTemplate::s(instruction_kind, rs1, rs2, imm));
    }

    pub(super) fn dispatch_address(
        &mut self,
        instruction_kind: impl Into<DispatchInstructionKind>,
        rs1: RegisterOperand,
        imm: i128,
    ) {
        self.dispatch(DispatchRowTemplate::address(instruction_kind, rs1, imm));
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

    fn dispatch(&mut self, row: DispatchRowTemplate) {
        self.ops.push(ExpansionOp::Dispatch(row));
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
    use jolt_riscv::{JoltInstructionKind, NormalizedInstruction, NormalizedOperands};

    use super::*;

    fn source() -> NormalizedInstruction {
        NormalizedInstruction {
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
