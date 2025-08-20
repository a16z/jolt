use crate::emulator::cpu::Cpu;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{
    normalize_register_value, InstructionFormat, InstructionRegisterState, NormalizedOperands,
};

/// `VirtualAssertHalfwordAlignment` is the only instruction that
/// uses `rs1` and `imm` but does not write to a destination register.
#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HalfwordAlignFormat {
    pub rs1: u8,
    pub imm: i64,
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct HalfwordAlignRegisterState {
    pub rs1: u64,
}

impl InstructionRegisterState for HalfwordAlignRegisterState {
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use rand::RngCore;
        Self {
            rs1: rng.next_u64(),
        }
    }

    fn rs1_value(&self) -> u64 {
        self.rs1
    }
}

impl InstructionFormat for HalfwordAlignFormat {
    type RegisterState = HalfwordAlignRegisterState;

    fn parse(_: u32) -> Self {
        unimplemented!("virtual instruction")
    }

    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rs1 = normalize_register_value(cpu.x[self.rs1 as usize], &cpu.xlen);
    }

    fn capture_post_execution_state(&self, _: &mut Self::RegisterState, _: &mut Cpu) {
        // No register write
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use common::constants::RISCV_REGISTER_COUNT;
        use rand::RngCore;
        Self {
            rs1: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
            imm: rng.next_u64() as i64,
        }
    }
}

impl From<NormalizedOperands> for HalfwordAlignFormat {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rs1: operands.rs1,
            imm: operands.imm as i64,
        }
    }
}

impl From<HalfwordAlignFormat> for NormalizedOperands {
    fn from(format: HalfwordAlignFormat) -> Self {
        Self {
            rs1: format.rs1,
            rs2: 0,
            rd: 0,
            imm: format.imm as i128,
        }
    }
}
