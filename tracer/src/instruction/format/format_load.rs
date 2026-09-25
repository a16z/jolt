use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{InstructionFormat, NormalizedOperands};

/// Same as FormatI, but with a signed `imm`. Used for load instructions,
/// which need to do signed field arithmetic with `imm` in R1CS constraints.
#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatLoad {
    pub rd: u8,
    pub rs1: u8,
    pub imm: i64,
}

impl InstructionFormat for FormatLoad {
    fn parse(word: u32) -> Self {
        FormatLoad {
            rd: ((word >> 7) & 0x1f) as u8,   // [11:7]
            rs1: ((word >> 15) & 0x1f) as u8, // [19:15]
            imm: (
                match word & 0x80000000 {
                    // imm[31:11] = [31]
                    0x80000000 => 0xfffff800,
                    _ => 0,
                } | ((word >> 20) & 0x000007ff)
                // imm[10:0] = [30:20]
            ) as i32 as i64,
        }
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use common::constants::RISCV_REGISTER_COUNT;
        use rand::RngCore;
        Self {
            // Keep imm small to avoid going out of bounds when added to rs1
            imm: (rng.next_u64() as i64 % 256) - 128, // Range: [-128, 127]
            rd: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
            // rs1 should never be 0 for memory operations (x0 is hardwired to 0)
            rs1: 1 + (rng.next_u64() as u8 % (RISCV_REGISTER_COUNT - 1)),
        }
    }

    fn set_rd(&mut self, rd: u8) {
        self.rd = rd;
    }
}

impl From<NormalizedOperands> for FormatLoad {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rd: operands.rd.unwrap(),
            rs1: operands.rs1.unwrap(),
            imm: operands.imm as i64,
        }
    }
}

impl From<FormatLoad> for NormalizedOperands {
    fn from(format: FormatLoad) -> Self {
        Self {
            rd: Some(format.rd),
            rs1: Some(format.rs1),
            rs2: None,
            imm: format.imm as i128,
        }
    }
}
