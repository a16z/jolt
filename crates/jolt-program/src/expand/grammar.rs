use jolt_riscv::{JoltInstructionKind, SourceInstructionKind, SourceInstructionRow};

use crate::expand::{
    allocator::NUM_VIRTUAL_INSTRUCTION_REGISTERS, operands::format_i_imm, ExpansionError,
};

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

impl From<TempId> for RegisterOperand {
    fn from(temp: TempId) -> Self {
        Self::Temp(temp)
    }
}

/// Symbolic inline-register placeholder, resolved to the inline virtual
/// register pool during materialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct InlineTempId(pub(super) u8);

impl InlineTempId {
    pub(super) const fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RegisterOperand {
    Register(u8),
    Temp(TempId),
    InlineTemp(InlineTempId),
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
pub(super) type SourceInstructionRowTemplate = InstructionTemplate<SourceInstructionKind>;

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
    Expand(SourceInstructionRowTemplate),
    Allocate(TempId),
    Release(TempId),
    AllocateInline(InlineTempId),
    ReleaseInline(InlineTempId),
}

/// A complete symbolic recipe for one source instruction.
///
/// The sequence still contains unresolved symbolic registers and possibly
/// source-only helper rows. It is not final bytecode until `ExpansionState`
/// binds registers, recursively expands helper rows, validates target legality,
/// and stamps sequence metadata.
pub struct ExpandedInstructionSequence {
    pub(super) source: SourceInstructionRow,
    pub(super) ops: Vec<ExpansionOp>,
}

/// Builds a symbolic expansion recipe from instruction/allocate/release calls.
pub(super) struct ExpansionBuilder {
    source: SourceInstructionRow,
    ops: Vec<ExpansionOp>,
    next_temp: usize,
}

impl ExpansionBuilder {
    pub(super) fn new(source: SourceInstructionRow) -> Self {
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

    pub(super) fn emit_r(
        &mut self,
        instruction_kind: impl Into<SourceInstructionKind>,
        rd: impl Into<RegisterOperand>,
        rs1: impl Into<RegisterOperand>,
        rs2: impl Into<RegisterOperand>,
    ) {
        self.instruction(SourceInstructionRowTemplate::r(
            instruction_kind.into(),
            rd.into(),
            rs1.into(),
            rs2.into(),
        ));
    }

    pub(super) fn emit_i(
        &mut self,
        instruction_kind: impl Into<SourceInstructionKind>,
        rd: impl Into<RegisterOperand>,
        rs1: impl Into<RegisterOperand>,
        imm: i128,
    ) {
        self.instruction(SourceInstructionRowTemplate::i(
            instruction_kind.into(),
            rd.into(),
            rs1.into(),
            imm,
        ));
    }

    pub(super) fn emit_j(
        &mut self,
        instruction_kind: impl Into<SourceInstructionKind>,
        rd: impl Into<RegisterOperand>,
        imm: i128,
    ) {
        self.instruction(SourceInstructionRowTemplate::j(
            instruction_kind.into(),
            rd.into(),
            imm,
        ));
    }

    pub(super) fn emit_u(
        &mut self,
        instruction_kind: impl Into<SourceInstructionKind>,
        rd: impl Into<RegisterOperand>,
        imm: i128,
    ) {
        self.instruction(SourceInstructionRowTemplate::u(
            instruction_kind.into(),
            rd.into(),
            imm,
        ));
    }

    pub(super) fn emit_b(
        &mut self,
        instruction_kind: impl Into<SourceInstructionKind>,
        rs1: impl Into<RegisterOperand>,
        rs2: impl Into<RegisterOperand>,
        imm: i128,
    ) {
        self.instruction(SourceInstructionRowTemplate::b(
            instruction_kind.into(),
            rs1.into(),
            rs2.into(),
            imm,
        ));
    }

    pub(super) fn emit_s(
        &mut self,
        instruction_kind: impl Into<SourceInstructionKind>,
        rs1: impl Into<RegisterOperand>,
        rs2: impl Into<RegisterOperand>,
        imm: i128,
    ) {
        self.instruction(SourceInstructionRowTemplate::s(
            instruction_kind.into(),
            rs1.into(),
            rs2.into(),
            imm,
        ));
    }

    pub(super) fn emit_ld(
        &mut self,
        instruction_kind: impl Into<SourceInstructionKind>,
        rd: impl Into<RegisterOperand>,
        rs1: impl Into<RegisterOperand>,
        imm: i128,
    ) {
        self.emit_i(instruction_kind, rd, rs1, imm);
    }

    /// Emit an address-form assert (`VirtualAssert{Word,Halfword}Alignment`).
    ///
    /// The offset is wrapped to `u64` exactly as `emit_i` callers do via
    /// [`format_i_imm`]. These asserts carry `AddOperands`, so the bytecode's
    /// `Imm` column is compared against a lookup index of `rs1 + imm` as a field
    /// element; a raw *signed* offset would leave the two disagreeing, and would
    /// make a negative effective address produce an index of `2^128 - |rs1+imm|`
    /// whose only satisfying representative sits in the fp128 alias band.
    pub(super) fn emit_address(
        &mut self,
        instruction_kind: impl Into<SourceInstructionKind>,
        rs1: impl Into<RegisterOperand>,
        imm: i128,
    ) {
        self.instruction(SourceInstructionRowTemplate::address(
            instruction_kind.into(),
            rs1.into(),
            format_i_imm(imm),
        ));
    }

    pub(super) fn release(&mut self, temp: TempId) {
        self.ops.push(ExpansionOp::Release(temp));
    }

    pub(super) fn allocate_inline(&mut self, temp: InlineTempId) {
        self.ops.push(ExpansionOp::AllocateInline(temp));
    }

    pub(super) fn release_inline(&mut self, temp: InlineTempId) {
        self.ops.push(ExpansionOp::ReleaseInline(temp));
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

    pub(super) fn emit(&mut self, row: RowTemplate) {
        self.ops.push(ExpansionOp::Emit(row));
    }

    fn instruction(&mut self, row: SourceInstructionRowTemplate) {
        if let Some(instruction_kind) = row.instruction_kind.jolt_kind() {
            self.emit(RowTemplate {
                instruction_kind,
                operands: row.operands,
            });
        } else {
            self.expand(row);
        }
    }

    fn expand(&mut self, row: SourceInstructionRowTemplate) {
        self.ops.push(ExpansionOp::Expand(row));
    }
}

/// Instructions that exist only in decoded source and must be expanded into target-legal sequences.
// Exposed (pub) so the Lean generator can gate on the same expand-vs-native
// decision the expander itself uses in `dispatch_source`.
pub fn is_source_only(instruction_kind: SourceInstructionKind) -> bool {
    matches!(
        instruction_kind,
        SourceInstructionKind::Inline
            | SourceInstructionKind::MULH
            | SourceInstructionKind::MULHSU
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
            | SourceInstructionKind::SLLIW
            | SourceInstructionKind::SLLW
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
    #![expect(
        clippy::panic_in_result_fn,
        reason = "test assertions inside Result-returning tests"
    )]

    use jolt_riscv::{NormalizedOperands, SourceInstructionRow};

    use super::*;

    fn source() -> SourceInstructionRow {
        SourceInstructionRow {
            address: 0x8000_0000,
            operands: NormalizedOperands::default(),
            inline: None,
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
