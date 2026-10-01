//! Operand format for unary register instructions.
//!
//! Guest assembly uses R-format syntax; the tracer retains only `rs1` and `rd`.
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{InstructionFormat, NormalizedOperands};

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatT {
    pub rd: u8,
    pub rs1: u8,
}

impl InstructionFormat for FormatT {
    fn parse(word: u32) -> Self {
        FormatT {
            rd: ((word >> 7) & 0x1f) as u8,   // [11:7]
            rs1: ((word >> 15) & 0x1f) as u8, // [19:15]
        }
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use common::constants::RISCV_REGISTER_COUNT;
        use rand::RngCore;
        Self {
            rd: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
            rs1: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
        }
    }

    fn set_rd(&mut self, rd: u8) {
        self.rd = rd;
    }
}

impl From<NormalizedOperands> for FormatT {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rd: operands.rd.unwrap(),
            rs1: operands.rs1.unwrap(),
        }
    }
}

impl From<FormatT> for NormalizedOperands {
    fn from(format: FormatT) -> Self {
        Self {
            rd: Some(format.rd),
            rs1: Some(format.rs1),
            rs2: None,
            imm: 0,
        }
    }
}
