use crate::emulator::cpu::Cpu;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{
    normalize_register_value, InstructionFormat, InstructionRegisterState, NormalizedOperands,
};

/// Format for assert instructions that use `rs1` and `imm` but do not write to a destination register.
///
/// Used by:
/// - `VirtualAssert` - asserts that rs1 is non-zero
/// - `VirtualAssertHalfwordAlignment` - asserts halfword alignment
/// - `VirtualAssertWordAlignment` - asserts word alignment
///
/// Note: Some assert instructions (like `VirtualAssertEQ`, `VirtualAssertLTE`) use two source
/// registers and therefore use `FormatB` instead.
#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatAssert {
    pub rs1: u8,
    pub imm: i64,
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateFormatAssert {
    pub rs1: u64,
}

impl InstructionRegisterState for RegisterStateFormatAssert {
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng, operands: &NormalizedOperands) -> Self {
        use rand::RngCore;
        Self {
            rs1: if operands.rs1.unwrap() == 0 {
                0
            } else {
                rng.next_u64()
            },
        }
    }

    fn rs1_value(&self) -> Option<u64> {
        Some(self.rs1)
    }
}

impl InstructionFormat for FormatAssert {
    type RegisterState = RegisterStateFormatAssert;

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

impl From<NormalizedOperands> for FormatAssert {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rs1: operands.rs1.unwrap(),
            imm: operands.imm as i64,
        }
    }
}

impl From<FormatAssert> for NormalizedOperands {
    fn from(format: FormatAssert) -> Self {
        Self {
            rs1: Some(format.rs1),
            rs2: None,
            rd: None,
            imm: format.imm as i128,
        }
    }
}
