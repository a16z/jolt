#[cfg(any(feature = "test-utils", test))]
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{InstructionFormat, NormalizedOperands};

/// Operands of the store-data shifters (`VirtualShiftData{B,H,W}`): `rd`
/// receives the store value in `rs1` shifted into the lane that the effective
/// address `rs2 + imm` selects.
///
/// `imm` is the signed store offset, as in `FormatS`: the shifter's right
/// instruction input is `Rs2Value + Imm`, which must equal the effective
/// address itself, a value below `2^64`.
#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatShiftData {
    pub rd: u8,
    pub rs1: u8,
    pub rs2: u8,
    pub imm: i64,
}

impl InstructionFormat for FormatShiftData {
    fn parse(_: u32) -> Self {
        unimplemented!("virtual instruction")
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut StdRng) -> Self {
        use common::constants::RISCV_REGISTER_COUNT;
        use rand::RngCore;
        Self {
            rd: rng.next_u64() as u8 % RISCV_REGISTER_COUNT,
            rs1: rng.next_u64() as u8 % RISCV_REGISTER_COUNT,
            // The base register must hold an address, so it cannot be x0.
            rs2: 1 + (rng.next_u64() as u8 % (RISCV_REGISTER_COUNT - 1)),
            imm: (rng.next_u64() as i64 % 256) - 128,
        }
    }

    fn set_rd(&mut self, rd: u8) {
        self.rd = rd;
    }
}

impl From<NormalizedOperands> for FormatShiftData {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rd: operands.rd.unwrap(),
            rs1: operands.rs1.unwrap(),
            rs2: operands.rs2.unwrap(),
            imm: operands.imm as i64,
        }
    }
}

impl From<FormatShiftData> for NormalizedOperands {
    fn from(format: FormatShiftData) -> Self {
        Self {
            rd: Some(format.rd),
            rs1: Some(format.rs1),
            rs2: Some(format.rs2),
            imm: format.imm as i128,
        }
    }
}
