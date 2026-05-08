use jolt_riscv::{JoltInstructionKind, NormalizedInstruction, NormalizedOperands};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExpansionOp {
    Row(RowTemplate),
    Expand(RowTemplate),
    Release(u8),
}

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
