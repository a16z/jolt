use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{InstructionFormat, NormalizedOperands};

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatVirtualRightShiftI<const MASK_WIDTH: usize = 64> {
    pub rd: u8,
    pub rs1: u8,
    pub imm: u64,
}

impl<const MASK_WIDTH: usize> InstructionFormat for FormatVirtualRightShiftI<MASK_WIDTH> {
    fn parse(_: u32) -> Self {
        unimplemented!("virtual instruction")
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use common::constants::RISCV_REGISTER_COUNT;
        use rand::RngCore;

        assert!((1..=64).contains(&MASK_WIDTH));
        let shift = rng.next_u64() % MASK_WIDTH as u64;
        let imm = ((1u128 << MASK_WIDTH) - (1u128 << shift)) as u64;

        Self {
            imm,
            rd: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
            rs1: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
        }
    }

    fn set_rd(&mut self, rd: u8) {
        self.rd = rd;
    }
}

impl<const MASK_WIDTH: usize> From<NormalizedOperands> for FormatVirtualRightShiftI<MASK_WIDTH> {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rd: operands.rd.unwrap(),
            rs1: operands.rs1.unwrap(),
            imm: operands.imm as u64,
        }
    }
}

impl<const MASK_WIDTH: usize> From<FormatVirtualRightShiftI<MASK_WIDTH>> for NormalizedOperands {
    fn from(format: FormatVirtualRightShiftI<MASK_WIDTH>) -> Self {
        Self {
            rd: Some(format.rd),
            rs1: Some(format.rs1),
            rs2: None,
            imm: format.imm as i128,
        }
    }
}
