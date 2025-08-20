use crate::emulator::cpu::Cpu;
use common::constants::RISCV_REGISTER_COUNT;
use rand::rngs::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{
    normalize_register_value, InstructionFormat, InstructionRegisterState, NormalizedOperands,
};

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatJ {
    pub rd: u8,
    pub imm: u64,
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateFormatJ {
    pub rd: (u64, u64), // (old_value, new_value)
}

impl InstructionRegisterState for RegisterStateFormatJ {
    fn random(rng: &mut StdRng) -> Self {
        Self {
            rd: (rng.next_u64(), rng.next_u64()),
        }
    }

    fn rd_values(&self) -> (u64, u64) {
        self.rd
    }
}

impl InstructionFormat for FormatJ {
    type RegisterState = RegisterStateFormatJ;

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

    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rd.0 = normalize_register_value(cpu.x[self.rd as usize], &cpu.xlen);
    }

    fn capture_post_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rd.1 = normalize_register_value(cpu.x[self.rd as usize], &cpu.xlen);
    }

    fn random(rng: &mut StdRng) -> Self {
        Self {
            rd: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
            imm: rng.next_u64(),
        }
    }
}

impl From<NormalizedOperands> for FormatJ {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rd: operands.rd,
            imm: operands.imm as u64,
        }
    }
}

impl From<FormatJ> for NormalizedOperands {
    fn from(format: FormatJ) -> Self {
        Self {
            rs1: 0,
            rs2: 0,
            rd: format.rd,
            imm: format.imm as i128,
        }
    }
}
