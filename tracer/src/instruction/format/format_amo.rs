use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{InstructionFormat, NormalizedOperands};

/// R-format operands for atomic memory operations.
#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatAMO {
    pub rd: u8,
    pub rs1: u8,
    pub rs2: u8,
}

impl InstructionFormat for FormatAMO {
    fn parse(word: u32) -> Self {
        FormatAMO {
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
            // rs1 should never be 0 for memory operations (x0 is hardwired to 0)
            rs1: 1 + (rng.next_u64() as u8 % (RISCV_REGISTER_COUNT - 1)),
            rs2: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
        }
    }

    fn set_rd(&mut self, rd: u8) {
        self.rd = rd;
    }
}

impl From<NormalizedOperands> for FormatAMO {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rd: operands.rd.unwrap(),
            rs1: operands.rs1.unwrap(),
            rs2: operands.rs2.unwrap(),
        }
    }
}

impl From<FormatAMO> for NormalizedOperands {
    fn from(format: FormatAMO) -> Self {
        Self {
            rd: Some(format.rd),
            rs1: Some(format.rs1),
            rs2: Some(format.rs2),
            imm: 0,
        }
    }
}
