use tracer::{
    emulator::cpu::Xlen,
    utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness},
};

use crate::{BLAKE2_FUNCT3, BLAKE2_FUNCT7, INLINE_OPCODE};

pub fn create_blake2_harness() -> InlineTestHarness {
    // Blake2 needs message block (128 bytes) + counter (8 bytes) + flag (8 bytes) contiguous at rs2
    // and state (64 bytes) at rs1
    let layout = InlineMemoryLayout::single_input(144, 64); // 144 bytes for message+params, 64-byte state
    InlineTestHarness::new(layout, Xlen::Bit64)
}

pub fn load_blake2_data(
    harness: &mut InlineTestHarness,
    state: &[u64; crate::STATE_VECTOR_LEN],
    message: &[u64; crate::MSG_BLOCK_LEN],
    counter: u64,
    is_final: bool,
) {
    harness.setup_registers(); // RS1=state, RS2=message+params
    harness.load_state64(state);

    // Blake2 expects message + counter + flag contiguously at rs2
    // Create combined input: message (16 u64s) + counter (1 u64) + flag (1 u64)
    let mut combined_input = Vec::with_capacity(18);
    combined_input.extend_from_slice(message);
    combined_input.push(counter);
    let flag_value = if is_final { 1u64 } else { 0u64 };
    combined_input.push(flag_value);

    // Load the combined input
    harness.load_input64(&combined_input);
}

pub fn read_state(harness: &mut InlineTestHarness) -> [u64; crate::STATE_VECTOR_LEN] {
    let vec = harness.read_output64(crate::STATE_VECTOR_LEN);
    let mut state = [0u64; crate::STATE_VECTOR_LEN];
    state.copy_from_slice(&vec);
    state
}

pub fn instruction() -> tracer::instruction::inline::INLINE {
    InlineTestHarness::create_default_instruction(INLINE_OPCODE, BLAKE2_FUNCT3, BLAKE2_FUNCT7)
}
