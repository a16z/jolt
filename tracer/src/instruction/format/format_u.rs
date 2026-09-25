use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{InstructionFormat, NormalizedOperands};

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatU {
    pub rd: u8,
    pub imm: u64,
}

impl InstructionFormat for FormatU {
    fn parse(word: u32) -> Self {
        FormatU {
            rd: ((word >> 7) & 0x1f) as u8, // [11:7]
            imm: (
                match word & 0x80000000 {
				0x80000000 => 0xffffffff00000000,
				_ => 0
			} | // imm[63:32] = [31]
			((word as u64) & 0xfffff000)
                // imm[31:12] = [31:12]
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

impl From<NormalizedOperands> for FormatU {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rd: operands.rd.unwrap(),
            imm: operands.imm as u64,
        }
    }
}

impl From<FormatU> for NormalizedOperands {
    fn from(format: FormatU) -> Self {
        Self {
            rs1: None,
            rs2: None,
            rd: Some(format.rd),
            imm: format.imm as i128,
        }
    }
}
