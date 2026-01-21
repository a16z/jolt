use crate::emulator::cpu::Cpu;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{
    normalize_register_value, InstructionFormat, InstructionRegisterState, NormalizedOperands,
};

/// FormatR specialized for AMO (Atomic Memory Operation) instructions
/// Uses the same format as FormatR but with a custom RegisterState that
/// constrains rs1 to be a valid memory address for testing
#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatAMO {
    pub rd: u8,
    pub rs1: u8,
    pub rs2: u8,
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateFormatRAMO {
    pub rd: (u64, u64), // (old_value, new_value)
    pub rs1: u64,       // Memory address - constrained to valid range
    pub rs2: u64,       // Value to use in atomic operation
}

impl InstructionRegisterState for RegisterStateFormatRAMO {
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng, operands: &NormalizedOperands) -> Self {
        use crate::instruction::test::{DRAM_BASE, TEST_MEMORY_CAPACITY};
        use rand::RngCore;

        // Ensure rs1 is a valid aligned address within memory bounds
        // AMO instructions require naturally aligned addresses
        // Addresses must be in DRAM region (starting at 0x80000000)
        let alignment = 8; // Align to 8 bytes for AMO.D, will work for AMO.W too
                           // Make sure we have a reasonable range for testing
        let max_offset = (TEST_MEMORY_CAPACITY / 2).min(0x10000) - alignment;
        let offset = (rng.next_u64() % (max_offset / alignment)) * alignment;
        let address = DRAM_BASE + offset;

        debug_assert_ne!(operands.rs1.unwrap(), 0);

        let rs2_value = match operands.rs2.unwrap() {
            0 => 0,
            _ if operands.rs2 == operands.rs1 => address,
            _ => rng.next_u64(),
        };

        Self {
            rd: (
                match operands.rd {
                    _ if operands.rd == operands.rs1 => address,
                    _ if operands.rd == operands.rs2 => rs2_value,
                    _ => rng.next_u64(),
                },
                rng.next_u64(),
            ),
            rs1: address,
            rs2: rs2_value,
        }
    }

    fn rs1_value(&self) -> Option<u64> {
        Some(self.rs1)
    }

    fn rs2_value(&self) -> Option<u64> {
        Some(self.rs2)
    }

    fn rd_values(&self) -> Option<(u64, u64)> {
        Some(self.rd)
    }
}

impl InstructionFormat for FormatAMO {
    type RegisterState = RegisterStateFormatRAMO;

    fn parse(word: u32) -> Self {
        FormatAMO {
            rd: ((word >> 7) & 0x1f) as u8,   // [11:7]
            rs1: ((word >> 15) & 0x1f) as u8, // [19:15]
            rs2: ((word >> 20) & 0x1f) as u8, // [24:20]
        }
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
            // rs1 should never be 0 for memory operations (x0 is hardwired to 0)
            rs1: 1 + (rng.next_u64() as u8 % (RISCV_REGISTER_COUNT - 1)),
            rs2: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
        }
    }
}

impl From<NormalizedOperands> for FormatAMO {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rd: operands.rd.unwrap(),
            rs1: operands.rs1.unwrap(),
            rs2: operands.rs2.unwrap(),
        }
    }
}

impl From<FormatAMO> for NormalizedOperands {
    fn from(format: FormatAMO) -> Self {
        Self {
            rd: Some(format.rd),
            rs1: Some(format.rs1),
            rs2: Some(format.rs2),
            imm: 0,
        }
    }
}
