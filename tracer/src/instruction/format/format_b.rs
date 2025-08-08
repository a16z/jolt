use crate::emulator::cpu::Cpu;
use common::constants::REGISTER_COUNT;
use rand::rngs::StdRng;
use rand::{Rng, RngCore};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{
    normalize_register_value, InstructionFormat, InstructionRegisterState, NormalizedOperands,
};

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatB {
    pub rs1: u8,
    pub rs2: u8,
    pub imm: i128,
}

impl From<NormalizedOperands> for FormatB {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rs1: operands.rs1,
            rs2: operands.rs2,
            imm: operands.imm,
        }
    }
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateFormatB {
    pub rs1: u64,
    pub rs2: u64,
}

impl InstructionRegisterState for RegisterStateFormatB {
    fn random(rng: &mut StdRng) -> Self {
        Self {
            rs1: rng.next_u64(),
            rs2: rng.next_u64(),
        }
    }

    fn rs1_value(&self) -> u64 {
        self.rs1
    }

    fn rs2_value(&self) -> u64 {
        self.rs2
    }
}

impl InstructionFormat for FormatB {
    type RegisterState = RegisterStateFormatB;

    fn parse(word: u32) -> Self {
        FormatB {
            rs1: ((word >> 15) & 0x1f) as u8, // [19:15]
            rs2: ((word >> 20) & 0x1f) as u8, // [24:20]
            imm: (
                match word & 0x80000000 { // imm[31:12] = [31]
				0x80000000 => 0xfffff000,
				_ => 0
			} |
			((word << 4) & 0x00000800) | // imm[11] = [7]
			((word >> 20) & 0x000007e0) | // imm[10:5] = [30:25]
			((word >> 7) & 0x0000001e)
                // imm[4:1] = [11:8]
            ) as i32 as i128,
        }
    }

    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rs1 = normalize_register_value(cpu.x[self.rs1 as usize], &cpu.xlen);
        state.rs2 = normalize_register_value(cpu.x[self.rs2 as usize], &cpu.xlen);
    }

    fn capture_post_execution_state(&self, _: &mut Self::RegisterState, _: &mut Cpu) {
        // No register write
    }

    fn random(rng: &mut StdRng) -> Self {
        Self {
            imm: rng.gen(),
            rs1: (rng.next_u64() as u8 % REGISTER_COUNT),
            rs2: (rng.next_u64() as u8 % REGISTER_COUNT),
        }
    }

    fn normalize(&self) -> NormalizedOperands {
        NormalizedOperands {
            rs1: self.rs1,
            rs2: self.rs2,
            rd: 0,
            imm: self.imm,
        }
    }
}
