use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

use crate::declare_riscv_instr;
use crate::emulator::cpu::Cpu;
use crate::instruction::format::format_r::FormatR;
use crate::instruction::format::InstructionFormat;
use crate::instruction::inline_blake2::{
    execute_blake2b_256, Blake2SequenceBuilder, NEEDED_REGISTERS,
};
use crate::instruction::{
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction, VirtualInstructionSequence,
};

declare_riscv_instr!(
    name   = BLAKE2,
    mask   = 0xfe00707f,  // Mask for funct7 + funct3 + opcode
    match  = 0x0400000b,  // funct7=0x02, funct3=0x0, opcode=0x0B (custom-0)
    format = FormatR,
    ram    = ()
);

impl BLAKE2 {
    fn exec(&self, cpu: &mut Cpu, _ram_access: &mut <BLAKE2 as RISCVInstruction>::RAMAccess) {
        // This is the "fast path" for emulation without tracing.
        // It performs the Blake2b compression using a native Rust implementation.

        // 1. Read the 8-word (64-byte) hash state from memory pointed to by rs1.
        let mut state = [0u64; 8];
        let state_addr = cpu.x[self.operands.rs1] as u64;
        for (i, word) in state.iter_mut().enumerate() {
            *word = cpu
                .mmu
                .load_doubleword(state_addr.wrapping_add((i * 8) as u64))
                .expect("BLAKE2: Failed to load state from memory")
                .0;
        }

        // 2. Read the 16-word (128-byte) message block from memory pointed to by rs2.
        let mut message_words = [0u64; 16];
        let block_addr = cpu.x[self.operands.rs2] as u64;
        for (i, word) in message_words.iter_mut().enumerate() {
            *word = cpu
                .mmu
                .load_doubleword(block_addr.wrapping_add((i * 8) as u64))
                .expect("BLAKE2: Failed to load message block from memory")
                .0;
        }

        // 3. Execute Blake2b compression function on the state with the message block.
        // For simplicity, we assume single-block operation
        let compressed_state = execute_blake2b_single_block(&state, &message_words);

        // 4. Write the compressed state back to memory.
        for (i, &word) in compressed_state.iter().enumerate() {
            cpu.mmu
                .store_doubleword(state_addr.wrapping_add((i * 8) as u64), word)
                .expect("BLAKE2: Failed to store state to memory");
        }
    }
}

impl VirtualInstructionSequence for BLAKE2 {
    fn virtual_sequence(&self) -> Vec<RV32IMInstruction> {
        let vr: [usize; NEEDED_REGISTERS] = core::array::from_fn(|i| virtual_register_index(i as u64) as usize);
        
        Blake2SequenceBuilder::new(
            self.address,
            vr,
            self.operands.rs1,
            self.operands.rs2,
        )
        .build()
    }
}

impl RISCVTrace for BLAKE2 {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        if let Some(trace) = trace {
            // Generate the virtual instruction sequence
            let virtual_sequence = self.virtual_sequence();
            
            // Execute each instruction in the sequence and add to trace
            for instruction in virtual_sequence {
                instruction.trace(cpu, Some(trace));
            }
        } else {
            // Fast path without tracing
            self.exec(cpu, &mut ());
        }
    }
}

/// Execute Blake2b compression for a single block (simplified version for instruction emulation)
fn execute_blake2b_single_block(state: &[u64; 8], message_words: &[u64; 16]) -> [u64; 8] {
    // Use the host implementation for compression
    let mut h = *state;
    compress_single_block(&mut h, message_words);
    h
}

/// Simplified Blake2b compression for single block
fn compress_single_block(h: &mut [u64; 8], m: &[u64; 16]) {
    use crate::instruction::inline_blake2::{BLAKE2B_IV, SIGMA};
    
    // Initialize working variables
    let mut v = [0u64; 16];
    v[0..8].copy_from_slice(h);
    v[8..16].copy_from_slice(&BLAKE2B_IV);
    
    // Assume single block operation for simplicity
    v[12] ^= 128; // Block size
    v[14] = !v[14]; // Final block flag
    
    // 12 rounds of mixing
    for round in 0..12 {
        let s = &SIGMA[round];
        
        // Column step
        g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        
        // Diagonal step  
        g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    
    // Finalize hash state
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

/// Blake2b G function
fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}