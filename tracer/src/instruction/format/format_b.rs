use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{InstructionFormat, NormalizedOperands};

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatB {
    pub rs1: u8,
    pub rs2: u8,
    pub imm: i128,
}

impl InstructionFormat for FormatB {
    fn parse(word: u32) -> Self {
        FormatB {
            rs1: ((word >> 15) & 0x1f) as u8, // [19:15]
            rs2: ((word >> 20) & 0x1f) as u8, // [24:20]
            imm: (
                match word & 0x80000000 { // imm[31:12] = [31]
				0x80000000 => 0xfffff000,
				_ => 0
			} |
			((word << 4) & 0x00000800) | // imm[11] = [7]
			((word >> 20) & 0x000007e0) | // imm[10:5] = [30:25]
			((word >> 7) & 0x0000001e)
                // imm[4:1] = [11:8]
            ) as i32 as i128,
        }
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use common::constants::RISCV_REGISTER_COUNT;
        use rand::{Rng, RngCore};
        Self {
            imm: rng.gen(),
            rs1: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
            rs2: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
        }
    }
}

impl From<NormalizedOperands> for FormatB {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rs1: operands.rs1.unwrap(),
            rs2: operands.rs2.unwrap(),
            imm: operands.imm,
        }
    }
}

impl From<FormatB> for NormalizedOperands {
    fn from(format: FormatB) -> Self {
        Self {
            rs1: Some(format.rs1),
            rs2: Some(format.rs2),
            rd: None,
            imm: format.imm,
        }
    }
}
