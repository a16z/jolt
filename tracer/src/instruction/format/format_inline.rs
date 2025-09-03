//! Custom format for Jolt inline operations (not a RISC-V instruction format).
//!
//! FormatInline writes results to memory pointed by `rs3` (or rs1/rs2), but never
//! modifies the register values themselves - only the memory they reference. This differs from
//! FormatR instructions which write to the destination register.
//!
//! Note: SDKs use FormatR in assembly code to be compatible with the `core::arch::asm` macro,
//! but are parsed as FormatInline instructions by the tracer.

use crate::emulator::cpu::Cpu;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::{
    normalize_register_value, InstructionFormat, InstructionRegisterState, NormalizedOperands,
};

#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormatInline {
    pub rs1: u8,
    pub rs2: u8,
    pub rs3: u8,
}

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegisterStateFormatInline {
    pub rs1: u64,
    pub rs2: u64,
    pub rs3: u64,
}

impl InstructionRegisterState for RegisterStateFormatInline {
    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use rand::RngCore;
        Self {
            rs1: rng.next_u64(),
            rs2: rng.next_u64(),
            rs3: rng.next_u64(),
        }
    }

    fn rs1_value(&self) -> u64 {
        self.rs1
    }

    fn rs2_value(&self) -> u64 {
        self.rs2
    }
}

impl InstructionFormat for FormatInline {
    type RegisterState = RegisterStateFormatInline;

    fn parse(word: u32) -> Self {
        FormatInline {
            rs3: ((word >> 7) & 0x1f) as u8,  // [11:7]
            rs1: ((word >> 15) & 0x1f) as u8, // [19:15]
            rs2: ((word >> 20) & 0x1f) as u8, // [24:20]
        }
    }

    fn capture_pre_execution_state(&self, state: &mut Self::RegisterState, cpu: &mut Cpu) {
        state.rs1 = normalize_register_value(cpu.x[self.rs1 as usize], &cpu.xlen);
        state.rs2 = normalize_register_value(cpu.x[self.rs2 as usize], &cpu.xlen);
        state.rs3 = normalize_register_value(cpu.x[self.rs3 as usize], &cpu.xlen);
    }

    fn capture_post_execution_state(&self, _state: &mut Self::RegisterState, _cpu: &mut Cpu) {
        // FormatInline doesn't modify any registers, so nothing to capture post-execution
    }

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        use common::constants::RISCV_REGISTER_COUNT;
        use rand::RngCore;
        Self {
            rs1: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
            rs2: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
            rs3: (rng.next_u64() as u8 % RISCV_REGISTER_COUNT),
        }
    }
}

impl From<NormalizedOperands> for FormatInline {
    fn from(operands: NormalizedOperands) -> Self {
        Self {
            rs1: operands.rs1,
            rs2: operands.rs2,
            rs3: operands.rd, // Map rd field to rs3
        }
    }
}

impl From<FormatInline> for NormalizedOperands {
    fn from(format: FormatInline) -> Self {
        Self {
            rs1: format.rs1,
            rs2: format.rs2,
            rd: format.rs3, // Map rs3 back to rd field
            imm: 0,
        }
    }
}
