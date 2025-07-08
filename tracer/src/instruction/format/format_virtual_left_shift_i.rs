use crate::emulator::cpu::Cpu;
use common::constants::REGISTER_COUNT;
use rand::rngs::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{normalize_register_value, InstructionFormat, NormalizedOperands};
// Reuse the RegisterState from right shift since it's identical
pub use super::format_virtual_right_shift_i::RegisterStateFormatVirtualI;

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatVirtualLeftShiftI {
    pub rd: usize,
    pub rs1: usize,
    pub imm: u64,
}

impl InstructionFormat for FormatVirtualLeftShiftI {
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
        // Both left and right rotations use trailing_zeros() to extract rotation amount,
        // so they need the same bitmask generation logic
        let imm = ones.wrapping_shl(64 - shift);
        Self {
            imm,
            rd: (rng.next_u64() % REGISTER_COUNT) as usize,
            rs1: (rng.next_u64() % REGISTER_COUNT) as usize,
        }
    }

    fn normalize(&self) -> NormalizedOperands {
        NormalizedOperands {
            rs1: self.rs1,
            rs2: 0,
            rd: self.rd,
            imm: self.imm as i64,
        }
    }
}
