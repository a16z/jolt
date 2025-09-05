use crate::trace_generator::execute_keccak_f;
use crate::{Keccak256State, NUM_LANES};
use tracer::{
    emulator::cpu::Cpu,
    instruction::{inline::INLINE, RISCVInstruction},
};

pub fn keccak256_exec(
    instr: &INLINE,
    cpu: &mut Cpu,
    _ram_access: &mut <INLINE as RISCVInstruction>::RAMAccess,
) {
    // This is the "fast path" for emulation without tracing.
    // It performs the Keccak permutation using a native Rust implementation.

    // 1. Read the 25-lane (200-byte) state from memory pointed to by rs1.
    let mut state: Keccak256State = [0u64; NUM_LANES];
    let base_addr = cpu.x[instr.operands.rs1 as usize] as u64;
    for (i, lane) in state.iter_mut().enumerate() {
        *lane = cpu
            .mmu
            .load_doubleword(base_addr.wrapping_add((i * 8) as u64))
            .expect("KECCAK256: Failed to load state from memory")
            .0;
    }

    // 2. Execute the Keccak-f permutation on the state.
    execute_keccak_f(&mut state);

    // 3. Write the permuted state back to memory.
    for (i, &lane) in state.iter().enumerate() {
        cpu.mmu
            .store_doubleword(base_addr.wrapping_add((i * 8) as u64), lane)
            .expect("KECCAK256: Failed to store state to memory");
    }
}
