use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{InstructionFormat, NormalizedOperands};

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatR {
    pub rd: u8,
    pub rs1: u8,
    pub rs2: u8,
}

impl InstructionFormat for FormatR {
    fn parse(word: u32) -> Self {
        FormatR {
            rd: ((word >> 7) & 0x1f) as u8,   // [11:7]
            rs1: ((word >> 15) & 0x1f) as u8, // [19:15]
            rs2: ((word >> 20) & 0x1f) as u8, // [24:20]
        }
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use common::constants::RISCV_REGISTER_COUNT;
        use rand::RngCore;
        Self {
            rd: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
            rs1: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
            rs2: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
        }
    }

    fn set_rd(&mut self, rd: u8) {
        self.rd = rd;
    }
}

impl From<NormalizedOperands> for FormatR {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rd: operands.rd.unwrap(),
            rs1: operands.rs1.unwrap(),
            rs2: operands.rs2.unwrap(),
        }
    }
}

impl From<FormatR> for NormalizedOperands {
    fn from(format: FormatR) -> Self {
        Self {
            rd: Some(format.rd),
            rs1: Some(format.rs1),
            rs2: Some(format.rs2),
            imm: 0,
        }
    }
}
