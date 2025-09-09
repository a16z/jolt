use crate::{INLINE_OPCODE, SHA256_FUNCT3, SHA256_FUNCT7, SHA256_INIT_FUNCT3, SHA256_INIT_FUNCT7};
use tracer::emulator::cpu::Xlen;
use tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};

pub type Sha256Block = [u32; 16];
pub type Sha256State = [u32; 8];

pub fn create_sha256_harness(xlen: Xlen) -> InlineTestHarness {
    // SHA256: rs1=state/output, rs2=input (same as Blake/Keccak)
    let layout = InlineMemoryLayout::single_input(64, 32); // 64-byte block, 32-byte state
    InlineTestHarness::new(layout, xlen)
}

pub fn instruction_sha256() -> tracer::instruction::inline::INLINE {
    InlineTestHarness::create_default_instruction(INLINE_OPCODE, SHA256_FUNCT3, SHA256_FUNCT7)
}

pub fn instruction_sha256init() -> tracer::instruction::inline::INLINE {
    InlineTestHarness::create_default_instruction(
        INLINE_OPCODE,
        SHA256_INIT_FUNCT3,
        SHA256_INIT_FUNCT7,
    )
}

pub mod sverify {
    use super::*;

    pub fn assert_states_equal(expected: &Sha256State, actual: &Sha256State, test_name: &str) {
        if expected != actual {
            println!("\n❌ {test_name} FAILED");
            println!("Expected state: {expected:08x?}");
            println!("Actual state:   {actual:08x?}");
            panic!("{test_name} failed: states do not match");
        }
    }
}
