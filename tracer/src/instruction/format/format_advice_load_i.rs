use crate::emulator::cpu::Cpu;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{
    normalize_register_value, InstructionFormat, InstructionRegisterState, NormalizedOperands,
};

/// Format for advice load instructions
/// ADVICE_LB, ADVICE_LH, ADVICE_LW, ADVICE_LD
/// Similar to FormatI but only uses rd (value comes from advice tape), not rs1 (address is implicit).
#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatAdviceLoadI {
    pub rd: u8,
    pub imm: u64,
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateFormatAdviceLoadI {
    pub rd: (u64, u64), // (old_value, new_value)
}

impl InstructionRegisterState for RegisterStateFormatAdviceLoadI {
    #[cfg(any(feature = "test-utils", test))]
    fn random(_rng: &mut rand::rngs::StdRng, _operands: &NormalizedOperands) -> Self {
        //use rand::RngCore;
        unimplemented!("random state generation for advice load instructions");
    }

    fn rd_values(&self) -> Option<(u64, u64)> {
        Some(self.rd)
    }
}

impl InstructionFormat for FormatAdviceLoadI {
    type RegisterState = RegisterStateFormatAdviceLoadI;

    fn parse(word: u32) -> Self {
        FormatAdviceLoadI {
            rd: ((word >> 7) & 0x1f) as u8,   // [11:7]
            imm: (word as i32 as i64) as u64, // sign-extended immediate from [31:20]
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
        use common::constants::{RISCV_REGISTER_COUNT, XLEN};
        use rand::RngCore;

        let mut imm = rng.next_u64();

        let (shift, len) = match XLEN {
            32 => (imm & 0x1f, 32),
            64 => (imm & 0x3f, 64),
            _ => panic!(),
        };
        let ones = (1u128 << (len - shift)) - 1;
        imm = (ones << shift) as u64;

        Self {
            imm,
            rd: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
        }
    }
}

impl From<NormalizedOperands> for FormatAdviceLoadI {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rd: operands.rd.unwrap(),
            imm: operands.imm as u64,
        }
    }
}

impl From<FormatAdviceLoadI> for NormalizedOperands {
    fn from(format: FormatAdviceLoadI) -> Self {
        Self {
            rd: Some(format.rd),
            rs1: None,
            rs2: None,
            imm: format.imm as i128,
        }
    }
}
