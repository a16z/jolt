//! Operand format for Jolt inline operations.
//!
//! The registers hold memory pointers. The R-format `rd` slot is named `rs3`
//! here; inline instructions write through these pointers without modifying
//! the registers.

use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{InstructionFormat, NormalizedOperands};

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatInline {
    pub rs1: u8,
    pub rs2: u8,
    pub rs3: u8,
}

impl InstructionFormat for FormatInline {
    fn parse(word: u32) -> Self {
        FormatInline {
            rs3: ((word >> 7) & 0x1f) as u8,  // [11:7]
            rs1: ((word >> 15) & 0x1f) as u8, // [19:15]
            rs2: ((word >> 20) & 0x1f) as u8, // [24:20]
        }
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use common::constants::RISCV_REGISTER_COUNT;
        use rand::RngCore;
        Self {
            rs1: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
            rs2: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
            rs3: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
        }
    }

    /// FormatInline maps rd ↔ rs3.
    fn set_rd(&mut self, rd: u8) {
        self.rs3 = rd;
    }
}

impl From<NormalizedOperands> for FormatInline {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rs1: operands.rs1.unwrap(),
            rs2: operands.rs2.unwrap(),
            rs3: operands.rd.unwrap(), // Map rd field to rs3
        }
    }
}

impl From<FormatInline> for NormalizedOperands {
    fn from(format: FormatInline) -> Self {
        Self {
            rs1: Some(format.rs1),
            rs2: Some(format.rs2),
            rd: Some(format.rs3), // Map rs3 back to rd field
            imm: 0,
        }
    }
}
