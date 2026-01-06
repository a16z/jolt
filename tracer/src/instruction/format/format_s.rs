use crate::emulator::cpu::Cpu;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{
    normalize_register_value, InstructionFormat, InstructionRegisterState, NormalizedOperands,
};

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatS {
    pub rs1: u8,
    pub rs2: u8,
    pub imm: i64,
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateFormatS {
    pub rs1: u64,
    pub rs2: u64,
}

impl InstructionRegisterState for RegisterStateFormatS {
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng, operands: &NormalizedOperands) -> Self {
        use crate::instruction::test::{DRAM_BASE, TEST_MEMORY_CAPACITY};
        use rand::RngCore;
        // Use a smaller range to avoid issues with boundaries
        let max_offset = (TEST_MEMORY_CAPACITY / 2).min(0x10000);
        let rs1_value = if operands.rs1.unwrap() == 0 {
            unreachable!()
        } else {
            DRAM_BASE + (rng.next_u64() % max_offset)
        };

        Self {
            rs1: rs1_value,
            rs2: if operands.rs2.unwrap() == 0 {
                0
            } else if operands.rs2 == operands.rs1 {
                rs1_value
            } else {
                rng.next_u64()
            },
        }
    }

    fn rs1_value(&self) -> Option<u64> {
        Some(self.rs1)
    }

    fn rs2_value(&self) -> Option<u64> {
        Some(self.rs2)
    }
}

impl InstructionFormat for FormatS {
    type RegisterState = RegisterStateFormatS;

    fn parse(word: u32) -> Self {
        FormatS {
            rs1: ((word >> 15) & 0x1f) as u8, // [19:15]
            rs2: ((word >> 20) & 0x1f) as u8, // [24:20]
            imm: (
                match word & 0x80000000 {
				0x80000000 => 0xfffff000,
				_ => 0
			} | // imm[31:12] = [31]
			((word >> 20) & 0xfe0) | // imm[11:5] = [31:25]
			((word >> 7) & 0x1f)
                // imm[4:0] = [11:7]
            ) as i32 as i64,
        }
    }

    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rs1 = normalize_register_value(cpu.x[self.rs1 as usize], &cpu.xlen);
        state.rs2 = normalize_register_value(cpu.x[self.rs2 as usize], &cpu.xlen);
    }

    fn capture_post_execution_state(&self, _: &mut Self::RegisterState, _: &mut Cpu) {
        // No register write
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use common::constants::RISCV_REGISTER_COUNT;
        use rand::RngCore;
        Self {
            // rs1 should never be 0 for memory operations (x0 is hardwired to 0)
            rs1: 1 + (rng.next_u64() as u8 % (RISCV_REGISTER_COUNT - 1)),
            rs2: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
            // Keep imm small to avoid going out of bounds when added to rs1
            imm: (rng.next_u64() as i64 % 256) - 128, // Range: [-128, 127]
        }
    }
}

impl From<NormalizedOperands> for FormatS {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rs1: operands.rs1.unwrap(),
            rs2: operands.rs2.unwrap(),
            imm: operands.imm as i64,
        }
    }
}

impl From<FormatS> for NormalizedOperands {
    fn from(format: FormatS) -> Self {
        Self {
            rs1: Some(format.rs1),
            rs2: Some(format.rs2),
            imm: format.imm as i128,
            rd: None,
        }
    }
}
