//! BLAKE2 execution implementation

use crate::trace_generator::{execute_blake2b_compression, HASH_STATE_SIZE, MESSAGE_BLOCK_SIZE};
use tracer::{
    emulator::cpu::Cpu,
    instruction::{inline::INLINE, RISCVInstruction},
};

/// Execute BLAKE2b compression inline instruction
pub fn blake2b_exec(
    instr: &INLINE,
    cpu: &mut Cpu,
    _ram_access: &mut <INLINE as RISCVInstruction>::RAMAccess,
) {
    // This is the "fast path" for emulation without tracing.
    // It performs the Blake2b compression using a native Rust implementation.

    // 1. Read the hash state (8 words) from memory pointed to by rs1
    let mut state = [0u64; HASH_STATE_SIZE];
    let state_addr = cpu.x[instr.operands.rs1 as usize] as u64;
    for (i, word) in state.iter_mut().enumerate() {
        *word = cpu
            .mmu
            .load_doubleword(state_addr.wrapping_add((i * 8) as u64))
            .expect("BLAKE2B: Failed to load state from memory")
            .0;
    }

    // 2. Read the message block (16 words) from memory pointed to by rs2
    let mut message_with_metadata = [0u64; MESSAGE_BLOCK_SIZE + 2];
    let message_addr = cpu.x[instr.operands.rs2 as usize] as u64;
    for (i, word) in message_with_metadata.iter_mut().enumerate() {
        *word = cpu
            .mmu
            .load_doubleword(message_addr.wrapping_add((i * 8) as u64))
            .expect("BLAKE2B: Failed to load message block from memory")
            .0;
    }

    // 3. Execute the Blake2b compression
    execute_blake2b_compression(&mut state, &message_with_metadata);

    // 4. Write the updated state back to memory
    for (i, &word) in state.iter().enumerate() {
        cpu.mmu
            .store_doubleword(state_addr.wrapping_add((i * 8) as u64), word)
            .expect("BLAKE2B: Failed to store state to memory");
    }
}