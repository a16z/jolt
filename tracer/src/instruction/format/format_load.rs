use crate::emulator::cpu::Cpu;
use common::constants::RISCV_REGISTER_COUNT;
use rand::rngs::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{
    normalize_register_value, InstructionFormat, InstructionRegisterState, NormalizedOperands,
};

/// Same as FormatI, but with a signed `imm`. Used for load instructions,
/// which need to do signed field arithmetic with `imm` in R1CS constraints.
#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatLoad {
    pub rd: u8,
    pub rs1: u8,
    pub imm: i64,
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateFormatLoad {
    pub rd: (u64, u64), // (old_value, new_value)
    pub rs1: u64,
}

impl InstructionRegisterState for RegisterStateFormatLoad {
    fn random(rng: &mut StdRng) -> Self {
        Self {
            rd: (rng.next_u64(), rng.next_u64()),
            rs1: rng.next_u64(),
        }
    }

    fn rs1_value(&self) -> u64 {
        self.rs1
    }

    fn rd_values(&self) -> (u64, u64) {
        self.rd
    }
}

impl InstructionFormat for FormatLoad {
    type RegisterState = RegisterStateFormatLoad;

    fn parse(word: u32) -> Self {
        FormatLoad {
            rd: ((word >> 7) & 0x1f) as u8,   // [11:7]
            rs1: ((word >> 15) & 0x1f) as u8, // [19:15]
            imm: (
                match word & 0x80000000 {
                    // imm[31:11] = [31]
                    0x80000000 => 0xfffff800,
                    _ => 0,
                } | ((word >> 20) & 0x000007ff)
                // imm[10:0] = [30:20]
            ) as i32 as i64,
        }
    }

    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rs1 = normalize_register_value(cpu.x[self.rs1 as usize], &cpu.xlen);
        state.rd.0 = normalize_register_value(cpu.x[self.rd as usize], &cpu.xlen);
    }

    fn capture_post_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rd.1 = normalize_register_value(cpu.x[self.rd as usize], &cpu.xlen);
    }

    fn random(rng: &mut StdRng) -> Self {
        Self {
            imm: rng.next_u64() as i64,
            rd: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
            rs1: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
        }
    }
}

impl From<NormalizedOperands> for FormatLoad {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rd: operands.rd,
            rs1: operands.rs1,
            imm: operands.imm as i64,
        }
    }
}

impl From<FormatLoad> for NormalizedOperands {
    fn from(format: FormatLoad) -> Self {
        Self {
            rd: format.rd,
            rs1: format.rs1,
            rs2: 0,
            imm: format.imm as i128,
        }
    }
}
