use crate::emulator::cpu::Cpu;
use common::constants::REGISTER_COUNT;
use rand::rngs::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{
    normalize_register_value, InstructionFormat, InstructionRegisterState, NormalizedOperands,
};

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatU {
    pub rd: usize,
    pub imm: u64,
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateFormatU {
    pub rd: (u64, u64), // (old_value, new_value)
}

impl InstructionRegisterState for RegisterStateFormatU {
    fn random(rng: &mut StdRng) -> Self {
        Self {
            rd: (rng.next_u64(), rng.next_u64()),
        }
    }

    fn rd_values(&self) -> (u64, u64) {
        self.rd
    }
}

impl InstructionFormat for FormatU {
    type RegisterState = RegisterStateFormatU;

    fn parse(word: u32) -> Self {
        FormatU {
            rd: ((word >> 7) & 0x1f) as usize, // [11:7]
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
        state.rd.0 = normalize_register_value(cpu.x[self.rd], &cpu.xlen);
    }

    fn capture_post_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rd.1 = normalize_register_value(cpu.x[self.rd], &cpu.xlen);
    }

    fn random(rng: &mut StdRng) -> Self {
        Self {
            rd: (rng.next_u64() % REGISTER_COUNT) as usize,
            imm: rng.next_u64(),
        }
    }

    fn normalize(&self) -> NormalizedOperands {
        NormalizedOperands {
            rs1: 0,
            rs2: 0,
            rd: self.rd,
            imm: self.imm as i64,
        }
    }
}
