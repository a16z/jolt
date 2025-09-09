use crate::{INLINE_OPCODE, SHA256_FUNCT3, SHA256_FUNCT7, SHA256_INIT_FUNCT3, SHA256_INIT_FUNCT7};
use tracer::emulator::cpu::Xlen;
use tracer::utils::inline_test_harness::{hash_helpers, InlineTestHarness};

pub type Sha256Block = [u32; 16];
pub type Sha256State = [u32; 8];

pub const RS1: u8 = 10;
pub const RS2: u8 = 11;

pub fn create_sha256_harness(xlen: Xlen) -> InlineTestHarness {
    hash_helpers::sha256_harness(xlen)
}

pub fn instruction_sha256() -> tracer::instruction::inline::INLINE {
    InlineTestHarness::create_instruction(INLINE_OPCODE, SHA256_FUNCT3, SHA256_FUNCT7, RS1, RS2, 0)
}

pub fn instruction_sha256init() -> tracer::instruction::inline::INLINE {
    InlineTestHarness::create_instruction(
        INLINE_OPCODE,
        SHA256_INIT_FUNCT3,
        SHA256_INIT_FUNCT7,
        RS1,
        RS2,
        0,
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
