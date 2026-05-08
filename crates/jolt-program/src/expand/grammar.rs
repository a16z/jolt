use jolt_riscv::{JoltInstructionKind, NormalizedOperands};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExpansionOp {
    Row(RowTemplate),
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
}
