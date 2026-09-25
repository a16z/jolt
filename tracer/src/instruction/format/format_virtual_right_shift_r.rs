use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{InstructionFormat, NormalizedOperands};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatVirtualRightShiftR<const MASK_WIDTH: usize = 64> {
    pub rd: u8,
    pub rs1: u8,
    pub rs2: u8,
}

impl<const MASK_WIDTH: usize> Default for FormatVirtualRightShiftR<MASK_WIDTH> {
    fn default() -> Self {
        Self {
            rd: 0,
            rs1: 1,
            rs2: 2,
        }
    }
}

impl<const MASK_WIDTH: usize> InstructionFormat for FormatVirtualRightShiftR<MASK_WIDTH> {
    fn parse(_: u32) -> Self {
        unimplemented!("virtual instruction")
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use common::constants::RISCV_REGISTER_COUNT;
        use rand::RngCore;
        let rd = rng.next_u64() as u8 % RISCV_REGISTER_COUNT;
        let rs1 = rng.next_u64() as u8 % RISCV_REGISTER_COUNT;

        // Ensure rs2 is non-zero and different from rs1
        let mut rs2 = 1 + (rng.next_u64() as u8 % (RISCV_REGISTER_COUNT - 1));
        if rs2 == rs1 {
            rs2 = if rs2 == RISCV_REGISTER_COUNT - 1 {
                1
            } else {
                rs2 + 1
            };
        }

        Self { rd, rs1, rs2 }
    }

    fn set_rd(&mut self, rd: u8) {
        self.rd = rd;
    }
}

impl<const MASK_WIDTH: usize> From<NormalizedOperands> for FormatVirtualRightShiftR<MASK_WIDTH> {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rd: operands.rd.unwrap(),
            rs1: operands.rs1.unwrap(),
            rs2: operands.rs2.unwrap(),
        }
    }
}

impl<const MASK_WIDTH: usize> From<FormatVirtualRightShiftR<MASK_WIDTH>> for NormalizedOperands {
    fn from(format: FormatVirtualRightShiftR<MASK_WIDTH>) -> Self {
        Self {
            rd: Some(format.rd),
            rs1: Some(format.rs1),
            rs2: Some(format.rs2),
            imm: 0,
        }
    }
}
