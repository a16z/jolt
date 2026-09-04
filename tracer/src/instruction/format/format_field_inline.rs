use crate::emulator::cpu::Cpu;
use serde::{Deserialize, Serialize};

use super::{
    normalize_register_value, InstructionFormat, InstructionRegisterState, NormalizedOperands,
};
use jolt_riscv::{FieldInlineOp, FieldInlineXRegisterRole};

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FormatFieldInline {
    pub op: Option<FieldInlineOp>,
    pub rd: Option<u8>,
    pub rs1: Option<u8>,
    pub rs2: Option<u8>,
    pub imm: i128,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisterStateFormatFieldInline {
    pub rs1: Option<u64>,
    pub rd_pre: Option<u64>,
    pub rd_post: Option<u64>,
}

impl InstructionRegisterState for RegisterStateFormatFieldInline {
    #[cfg(any(feature = "test-utils", test))]
    fn random(_rng: &mut rand::rngs::StdRng, _operands: &NormalizedOperands) -> Self {
        Self::default()
    }

    fn rs1_value(&self) -> Option<u64> {
        self.rs1
    }

    fn rd_values(&self) -> Option<(u64, u64)> {
        self.rd_pre.zip(self.rd_post)
    }
}

impl InstructionFormat for FormatFieldInline {
    type RegisterState = RegisterStateFormatFieldInline;

    fn parse(word: u32) -> Self {
        let op = FieldInlineOp::from_word(word);
        let rd = ((word >> 7) & 0x1f) as u8;
        let rs1 = ((word >> 15) & 0x1f) as u8;
        let rs2 = ((word >> 20) & 0x1f) as u8;
        match op {
            Some(FieldInlineOp::Add | FieldInlineOp::Sub | FieldInlineOp::Mul) => Self {
                op,
                rd: Some(rd),
                rs1: Some(rs1),
                rs2: Some(rs2),
                imm: 0,
            },
            Some(FieldInlineOp::Inv | FieldInlineOp::LoadFromX | FieldInlineOp::StoreToX) => Self {
                op,
                rd: Some(rd),
                rs1: Some(rs1),
                rs2: None,
                imm: 0,
            },
            Some(FieldInlineOp::AssertEq) => Self {
                op,
                rd: None,
                rs1: Some(rs1),
                rs2: Some(rs2),
                imm: 0,
            },
            Some(FieldInlineOp::LoadImm) => Self {
                op,
                rd: Some(rd),
                rs1: None,
                rs2: None,
                imm: i128::from((word >> 20) & 0xfff),
            },
            None => Self::default(),
        }
    }

    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        if matches!(
            self.x_register_role(),
            Some(FieldInlineXRegisterRole::ReadRs1)
        ) {
            if let Some(rs1) = self.rs1 {
                state.rs1 = Some(normalize_register_value(cpu, rs1 as usize));
            }
        }
        if matches!(
            self.x_register_role(),
            Some(FieldInlineXRegisterRole::WriteRd)
        ) {
            if let Some(rd) = self.rd {
                state.rd_pre = Some(normalize_register_value(cpu, rd as usize));
            }
        }
    }

    fn capture_post_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        if matches!(
            self.x_register_role(),
            Some(FieldInlineXRegisterRole::WriteRd)
        ) {
            if let Some(rd) = self.rd {
                state.rd_post = Some(normalize_register_value(cpu, rd as usize));
            }
        }
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(_rng: &mut rand::rngs::StdRng) -> Self {
        Self::default()
    }

    fn set_rd(&mut self, rd: u8) {
        self.rd = Some(rd);
    }
}

impl FormatFieldInline {
    fn x_register_role(self) -> Option<FieldInlineXRegisterRole> {
        self.op
            .and_then(|op| jolt_riscv::field_inline_operand_shape_for_op(op).bridge_x_register_role)
    }
}

impl From<NormalizedOperands> for FormatFieldInline {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            op: None,
            rd: operands.rd,
            rs1: operands.rs1,
            rs2: operands.rs2,
            imm: operands.imm,
        }
    }
}

impl From<FormatFieldInline> for NormalizedOperands {
    fn from(format: FormatFieldInline) -> Self {
        Self {
            rd: format.rd,
            rs1: format.rs1,
            rs2: format.rs2,
            imm: format.imm,
        }
    }
}
