use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{InstructionFormat, NormalizedOperands};

/// Format for advice load instructions
/// ADVICE_LB, ADVICE_LH, ADVICE_LW, ADVICE_LD
/// Similar to FormatI but only uses rd (value comes from advice tape), not rs1 (address is implicit).
#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatAdviceLoadI {
    pub rd: u8,
}

impl InstructionFormat for FormatAdviceLoadI {
    fn parse(word: u32) -> Self {
        FormatAdviceLoadI {
            rd: ((word >> 7) & 0x1f) as u8, // [11:7]
        }
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use common::constants::RISCV_REGISTER_COUNT;
        use rand::RngCore;

        Self {
            rd: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
        }
    }

    fn set_rd(&mut self, rd: u8) {
        self.rd = rd;
    }
}

impl From<NormalizedOperands> for FormatAdviceLoadI {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rd: operands.rd.unwrap(),
        }
    }
}

impl From<FormatAdviceLoadI> for NormalizedOperands {
    fn from(format: FormatAdviceLoadI) -> Self {
        Self {
            rd: Some(format.rd),
            rs1: None,
            rs2: None,
            imm: 0,
        }
    }
}
