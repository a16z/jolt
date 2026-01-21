use crate::emulator::cpu::Cpu;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{
    normalize_register_value, InstructionFormat, InstructionRegisterState, NormalizedOperands,
};

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatU {
    pub rd: u8,
    pub imm: u64,
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateFormatU {
    pub rd: (u64, u64), // (old_value, new_value)
}

impl InstructionRegisterState for RegisterStateFormatU {
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng, _operands: &NormalizedOperands) -> Self {
        use rand::RngCore;
        Self {
            rd: (rng.next_u64(), rng.next_u64()),
        }
    }

    fn rd_values(&self) -> Option<(u64, u64)> {
        Some(self.rd)
    }
}

impl InstructionFormat for FormatU {
    type RegisterState = RegisterStateFormatU;

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

    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rd.0 = normalize_register_value(cpu.x[self.rd as usize], &cpu.xlen);
    }

    fn capture_post_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rd.1 = normalize_register_value(cpu.x[self.rd as usize], &cpu.xlen);
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
