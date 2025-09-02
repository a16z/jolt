//! FormatInline - Custom instruction format for Jolt inline operations
//!
//! This is not a RISC-V instruction format, but rather a custom format defined by the Jolt team
//! specifically for inline operations.
//!
//! ## Requirements for Inlines
//!
//! Inline operations have two key requirements:
//! 1. **Risc-V registers must not be changed at all** - Unlike regular instructions, inline operations
//!    cannot modify any risc-v registers, including the destination register (rd).
//! 2. **Memory operations are allowed** - Inlines can read from addresses pointed to by registers
//!    and write their results back to memory addresses. While registers remain unchanged, memory
//!    locations can be modified.
//!
//! ## Memory Operations with rs3
//!
//! FormatInline uses `rs3` to specify where the result of the inline operation should be written.
//! Instead of writing a result back to the `rd` register (as traditional instructions do), the
//! result is written to the memory location that `rs3` points to. However, this is not a strict
//! requirement - an inline operation might alternatively write its result to memory locations
//! pointed to by `rs1` or `rs2`, depending on the specific inline's implementation.
//!
//! ## Relationship to FormatR
//!
//! FormatInline is very similar to FormatR instructions, with one key difference: we use `rs3`
//! instead of `rd`. This reflects the fact that inline operations don't have a destination
//! register in the traditional sense.
//!
//! In program SDKs where inline assembly is called, to avoid introducing a new format to the
//! `asm!` macro, FormatR is still used for inline instructions in the assembly code. However,
//! these instructions are parsed as FormatInline instructions by the tracer.

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
