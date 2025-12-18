use crate::emulator::cpu::Cpu;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{
    normalize_register_value, InstructionFormat, InstructionRegisterState, NormalizedOperands,
};

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatVirtualRightShiftI {
    pub rd: u8,
    pub rs1: u8,
    pub imm: u64,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateFormatVirtualI {
    pub rd: (u64, u64), // (old_value, new_value)
    pub rs1: u64,
}

impl Default for RegisterStateFormatVirtualI {
    fn default() -> Self {
        Self { rd: (0, 0), rs1: 1 }
    }
}

impl InstructionRegisterState for RegisterStateFormatVirtualI {
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng, operands: &NormalizedOperands) -> Self {
        use rand::RngCore;
        let rs1_value = if operands.rs1.unwrap() == 0 {
            0
        } else {
            rng.next_u64()
        };

        Self {
            rd: (
                if operands.rd == operands.rs1 {
                    rs1_value
                } else {
                    rng.next_u64()
                },
                rng.next_u64(),
            ),
            rs1: rs1_value,
        }
    }

    fn rs1_value(&self) -> Option<u64> {
        Some(self.rs1)
    }

    fn rd_values(&self) -> Option<(u64, u64)> {
        Some(self.rd)
    }
}

impl InstructionFormat for FormatVirtualRightShiftI {
    type RegisterState = RegisterStateFormatVirtualI;

    fn parse(_: u32) -> Self {
        unimplemented!("virtual instruction")
    }

    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rs1 = normalize_register_value(cpu.x[self.rs1 as usize], &cpu.xlen);
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
            rs1: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
        }
    }
}

impl From<NormalizedOperands> for FormatVirtualRightShiftI {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rd: operands.rd.unwrap(),
            rs1: operands.rs1.unwrap(),
            imm: operands.imm as u64,
        }
    }
}

impl From<FormatVirtualRightShiftI> for NormalizedOperands {
    fn from(format: FormatVirtualRightShiftI) -> Self {
        Self {
            rd: Some(format.rd),
            rs1: Some(format.rs1),
            rs2: None,
            imm: format.imm as i128,
        }
    }
}
