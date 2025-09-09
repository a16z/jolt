use crate::{BLAKE2_FUNCT3, BLAKE2_FUNCT7, INLINE_OPCODE};
use tracer::emulator::cpu::Xlen;
use tracer::utils::inline_test_harness::InlineTestHarness;

pub const RS1: u8 = 10;
pub const RS2: u8 = 11;

pub fn create_blake2_harness() -> InlineTestHarness {
    tracer::utils::inline_test_harness::hash_helpers::blake2_harness(Xlen::Bit64)
}

pub fn load_blake2_data(
    harness: &mut InlineTestHarness,
    state: &[u64; crate::STATE_VECTOR_LEN],
    message: &[u64; crate::MSG_BLOCK_LEN],
    counter: u64,
    is_final: bool,
) {
    harness.setup_registers(RS2, RS1, None); // Note: Blake2 has swapped order (message first, state second)
    harness.load_state64(state);
    harness.load_input64(message);
    // Load counter and flag as second input
    let flag_value = if is_final { 1u64 } else { 0u64 };
    harness.load_input2_64(&[counter, flag_value]);
}

pub fn read_state(harness: &mut InlineTestHarness) -> [u64; crate::STATE_VECTOR_LEN] {
    let vec = harness.read_output64(crate::STATE_VECTOR_LEN);
    let mut state = [0u64; crate::STATE_VECTOR_LEN];
    state.copy_from_slice(&vec);
    state
}

pub fn instruction() -> tracer::instruction::inline::INLINE {
    InlineTestHarness::create_instruction(INLINE_OPCODE, BLAKE2_FUNCT3, BLAKE2_FUNCT7, RS1, RS2, 0)
}
