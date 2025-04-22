use crate::emulator::cpu::Cpu;
use common::constants::REGISTER_COUNT;
use rand::rngs::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{normalize_register_value, InstructionFormat, InstructionRegisterState};

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatI {
    pub rd: usize,
    pub rs1: usize,
    pub imm: i64,
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize)]
pub struct RegisterStateFormatI {
    pub rd: (u64, u64), // (old_value, new_value)
    pub rs1: u64,
}

impl InstructionRegisterState for RegisterStateFormatI {
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

impl InstructionFormat for FormatI {
    type RegisterState = RegisterStateFormatI;

    fn parse(word: u32) -> Self {
        FormatI {
            rd: ((word >> 7) & 0x1f) as usize,   // [11:7]
            rs1: ((word >> 15) & 0x1f) as usize, // [19:15]
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
        state.rs1 = normalize_register_value(cpu.x[self.rs1], &cpu.xlen);
        state.rd.0 = normalize_register_value(cpu.x[self.rd], &cpu.xlen);
    }

    fn capture_post_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rd.1 = normalize_register_value(cpu.x[self.rd], &cpu.xlen);
    }

    fn random(rng: &mut StdRng) -> Self {
        Self {
            imm: rng.next_u64() as i64,
            rd: (rng.next_u64() % REGISTER_COUNT) as usize,
            rs1: (rng.next_u64() % REGISTER_COUNT) as usize,
        }
    }

    fn rs1(&self) -> usize {
        self.rs1
    }

    fn rd(&self) -> usize {
        self.rd
    }
}
