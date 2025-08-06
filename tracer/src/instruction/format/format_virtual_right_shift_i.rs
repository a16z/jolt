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
pub struct FormatVirtualRightShiftI {
    pub rd: usize,
    pub rs1: usize,
    pub imm: u64,
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateFormatVirtualI {
    pub rd: (u64, u64), // (old_value, new_value)
    pub rs1: u64,
}

impl InstructionRegisterState for RegisterStateFormatVirtualI {
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

impl InstructionFormat for FormatVirtualRightShiftI {
    type RegisterState = RegisterStateFormatVirtualI;

    fn parse(_: u32) -> Self {
        unimplemented!("virtual instruction")
    }

    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rs1 = normalize_register_value(cpu.x[self.rs1], &cpu.xlen);
        state.rd.0 = normalize_register_value(cpu.x[self.rd], &cpu.xlen);
    }

    fn capture_post_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rd.1 = normalize_register_value(cpu.x[self.rd], &cpu.xlen);
    }

    fn random(rng: &mut StdRng) -> Self {
        let shift = rng.next_u32() % 64;
        let ones: u64 = (1 << shift) - 1;
        let imm = ones.wrapping_shl(64 - shift);
        Self {
            imm,
            rd: (rng.next_u64() % REGISTER_COUNT) as usize,
            rs1: (rng.next_u64() % REGISTER_COUNT) as usize,
        }
    }

    fn from_normalized(operands: NormalizedOperands) -> Self {
        Self {
            rd: operands.rd,
            rs1: operands.rs1,
            imm: operands.imm as u64,
        }
    }

    fn normalize(&self) -> NormalizedOperands {
        NormalizedOperands {
            rs1: self.rs1,
            rs2: 0,
            rd: self.rd,
            imm: self.imm as i128,
        }
    }
}
