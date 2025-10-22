use crate::emulator::cpu::Cpu;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{
    normalize_register_value, InstructionFormat, InstructionRegisterState, NormalizedOperands,
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatVirtualRightShiftR {
    pub rd: u8,
    pub rs1: u8,
    pub rs2: u8,
}

impl Default for FormatVirtualRightShiftR {
    fn default() -> Self {
        Self {
            rd: 0,
            rs1: 1, // Default to 1 instead of 0
            rs2: 2,
        }
    }
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateVirtualRightShift {
    pub rd: (u64, u64), // (old_value, new_value)
    pub rs1: u64,
    pub rs2: u64,
}

impl InstructionRegisterState for RegisterStateVirtualRightShift {
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng, operands: &NormalizedOperands) -> Self {
        use rand::RngCore;
        let rs1_value = if operands.rs1 == 0 { 0 } else { rng.next_u64() };

        let shift = rng.next_u32() % 64;
        let ones: u64 = (1 << shift) - 1;
        let rs2_value = ones.wrapping_shl(64 - shift);

        Self {
            rd: (rng.next_u64(), rng.next_u64()),
            rs1: rs1_value,
            rs2: if operands.rs2 == 0 {
                panic!()
            }
            // else if operands.rs2 == operands.rs1 {
            //     rs1_value
            // }
            else {
                rs2_value
            },
        }
    }

    fn rs1_value(&self) -> u64 {
        self.rs1
    }

    fn rs2_value(&self) -> u64 {
        self.rs2
    }

    fn rd_values(&self) -> (u64, u64) {
        self.rd
    }
}

impl InstructionFormat for FormatVirtualRightShiftR {
    type RegisterState = RegisterStateVirtualRightShift;

    fn parse(_: u32) -> Self {
        unimplemented!("virtual instruction")
    }

    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rs1 = normalize_register_value(cpu.x[self.rs1 as usize], &cpu.xlen);
        state.rs2 = normalize_register_value(cpu.x[self.rs2 as usize], &cpu.xlen);
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
            rs1: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
            rs2: 1 + (rng.next_u64() as u8 % (RISCV_REGISTER_COUNT - 1)),
        }
    }
}

impl From<NormalizedOperands> for FormatVirtualRightShiftR {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rd: operands.rd,
            rs1: operands.rs1,
            rs2: operands.rs2,
        }
    }
}

impl From<FormatVirtualRightShiftR> for NormalizedOperands {
    fn from(format: FormatVirtualRightShiftR) -> Self {
        Self {
            rd: format.rd,
            rs1: format.rs1,
            rs2: format.rs2,
            imm: 0,
        }
    }
}
