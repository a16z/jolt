use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{InstructionFormat, NormalizedOperands};

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatS {
    pub rs1: u8,
    pub rs2: u8,
    pub imm: i64,
}

impl InstructionFormat for FormatS {
    fn parse(word: u32) -> Self {
        FormatS {
            rs1: ((word >> 15) & 0x1f) as u8, // [19:15]
            rs2: ((word >> 20) & 0x1f) as u8, // [24:20]
            imm: (
                match word & 0x80000000 {
				0x80000000 => 0xfffff000,
				_ => 0
			} | // imm[31:12] = [31]
			((word >> 20) & 0xfe0) | // imm[11:5] = [31:25]
			((word >> 7) & 0x1f)
                // imm[4:0] = [11:7]
            ) as i32 as i64,
        }
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use common::constants::RISCV_REGISTER_COUNT;
        use rand::RngCore;
        Self {
            // rs1 should never be 0 for memory operations (x0 is hardwired to 0)
            rs1: 1 + (rng.next_u64() as u8 % (RISCV_REGISTER_COUNT - 1)),
            rs2: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
            // Keep imm small to avoid going out of bounds when added to rs1
            imm: (rng.next_u64() as i64 % 256) - 128, // Range: [-128, 127]
        }
    }
}

impl From<NormalizedOperands> for FormatS {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rs1: operands.rs1.unwrap(),
            rs2: operands.rs2.unwrap(),
            imm: operands.imm as i64,
        }
    }
}

impl From<FormatS> for NormalizedOperands {
    fn from(format: FormatS) -> Self {
        Self {
            rs1: Some(format.rs1),
            rs2: Some(format.rs2),
            imm: format.imm as i128,
            rd: None,
        }
    }
}
