use jolt_riscv::{JoltInstructionKind, NormalizedInstruction};

use crate::expand::ExpansionError;

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

#[derive(Clone, Copy)]
pub(super) enum ExpansionOp {
    Emit(RowTemplate),
    Expand(RowTemplate),
    Allocate(TempId),
    Release(TempId),
}

pub(super) struct ExpandedInstructionSequence {
    pub(super) source: NormalizedInstruction,
    pub(super) ops: Vec<ExpansionOp>,
}

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
        if self.next_temp >= 256 {
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
    ) -> Result<(), ExpansionError> {
        self.expand(RowTemplate::r(instruction_kind, rd, rs1, rs2))
    }

    pub(super) fn expand_i(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rd: RegisterOperand,
        rs1: RegisterOperand,
        imm: i128,
    ) -> Result<(), ExpansionError> {
        self.expand(RowTemplate::i(instruction_kind, rd, rs1, imm))
    }

    pub(super) fn expand_j(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rd: RegisterOperand,
        imm: i128,
    ) -> Result<(), ExpansionError> {
        self.expand(RowTemplate::j(instruction_kind, rd, imm))
    }

    pub(super) fn expand_u(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rd: RegisterOperand,
        imm: i128,
    ) -> Result<(), ExpansionError> {
        self.expand(RowTemplate::u(instruction_kind, rd, imm))
    }

    pub(super) fn expand_b(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rs1: RegisterOperand,
        rs2: RegisterOperand,
        imm: i128,
    ) -> Result<(), ExpansionError> {
        self.expand(RowTemplate::b(instruction_kind, rs1, rs2, imm))
    }

    pub(super) fn expand_s(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rs1: RegisterOperand,
        rs2: RegisterOperand,
        imm: i128,
    ) -> Result<(), ExpansionError> {
        self.expand(RowTemplate::s(instruction_kind, rs1, rs2, imm))
    }

    pub(super) fn expand_address(
        &mut self,
        instruction_kind: JoltInstructionKind,
        rs1: RegisterOperand,
        imm: i128,
    ) -> Result<(), ExpansionError> {
        self.expand(RowTemplate::address(instruction_kind, rs1, imm))
    }

    pub(super) fn release(&mut self, temp: TempId) -> Result<(), ExpansionError> {
        self.ops.push(ExpansionOp::Release(temp));
        Ok(())
    }

    pub(super) fn release_many<const N: usize>(
        &mut self,
        registers: [TempId; N],
    ) -> Result<(), ExpansionError> {
        let mut index = 0;
        while index < N {
            self.release(registers[index])?;
            index += 1;
        }
        Ok(())
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

    fn expand(&mut self, row: RowTemplate) -> Result<(), ExpansionError> {
        self.ops.push(ExpansionOp::Expand(row));
        Ok(())
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
