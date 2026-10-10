use serde::{Deserialize, Serialize};

use super::{InstructionFormat, NormalizedOperands};
use jolt_riscv::{field_inline_load_accumulate_from_memory_offset, FieldInlineOp};

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FormatFieldInline {
    pub rd: Option<u8>,
    pub rs1: Option<u8>,
    pub rs2: Option<u8>,
    pub imm: i128,
}

impl InstructionFormat for FormatFieldInline {
    fn parse(word: u32) -> Self {
        let op = FieldInlineOp::from_word(word);
        let rd = ((word >> 7) & 0x1f) as u8;
        let rs1 = ((word >> 15) & 0x1f) as u8;
        let rs2 = ((word >> 20) & 0x1f) as u8;
        match op {
            Some(FieldInlineOp::Add | FieldInlineOp::Sub | FieldInlineOp::Mul) => Self {
                rd: Some(rd),
                rs1: Some(rs1),
                rs2: Some(rs2),
                imm: 0,
            },
            Some(FieldInlineOp::Inv | FieldInlineOp::LoadAccumulateFromRegister) => Self {
                rd: Some(rd),
                rs1: Some(rs1),
                rs2: None,
                imm: 0,
            },
            Some(FieldInlineOp::AssertEq) => Self {
                rd: None,
                rs1: Some(rs1),
                rs2: Some(rs2),
                imm: 0,
            },
            Some(FieldInlineOp::AssertZero) => Self {
                rd: None,
                rs1: Some(rs1),
                rs2: None,
                imm: 0,
            },
            Some(FieldInlineOp::LoadImm) => Self {
                rd: Some(rd),
                rs1: None,
                rs2: None,
                imm: i128::from((word >> 20) & 0xfff),
            },
            // `rd` is the integer destination, `rs1` the x base, `rs2` the
            // field destination; the word offset rides funct7 as the imm.
            Some(FieldInlineOp::LoadAccumulateFromMemory) => Self {
                rd: Some(rd),
                rs1: Some(rs1),
                rs2: Some(rs2),
                imm: i128::from(field_inline_load_accumulate_from_memory_offset(word)),
            },
            // `rd` is the x-register taking the low limb, `rs1` the field
            // source, `rs2` the field register taking the quotient.
            Some(FieldInlineOp::AdviceLimb) => Self {
                rd: Some(rd),
                rs1: Some(rs1),
                rs2: Some(rs2),
                imm: 0,
            },
            None => Self::default(),
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

impl From<NormalizedOperands> for FormatFieldInline {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
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
