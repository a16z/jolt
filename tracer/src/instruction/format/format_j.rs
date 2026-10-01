use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{InstructionFormat, NormalizedOperands};

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatJ {
    pub rd: u8,
    pub imm: u64,
}

impl InstructionFormat for FormatJ {
    fn parse(word: u32) -> Self {
        FormatJ {
            rd: ((word >> 7) & 0x1f) as u8, // [11:7]
            imm: (
                match word & 0x80000000 { // imm[31:20] = [31]
				0x80000000 => 0xfff00000,
				_ => 0
			} |
			(word & 0x000ff000) | // imm[19:12] = [19:12]
			((word & 0x00100000) >> 9) | // imm[11] = [20]
			((word & 0x7fe00000) >> 20)
                // imm[10:1] = [30:21]
            ) as i32 as i64 as u64,
        }
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use common::constants::RISCV_REGISTER_COUNT;
        use rand::RngCore;
        Self {
            rd: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
            imm: rng.next_u64(),
        }
    }

    fn set_rd(&mut self, rd: u8) {
        self.rd = rd;
    }
}

impl From<NormalizedOperands> for FormatJ {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rd: operands.rd.unwrap(),
            imm: operands.imm as u64,
        }
    }
}

impl From<FormatJ> for NormalizedOperands {
    fn from(format: FormatJ) -> Self {
        Self {
            rs1: None,
            rs2: None,
            rd: Some(format.rd),
            imm: format.imm as i128,
        }
    }
}
