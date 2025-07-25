use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

use crate::declare_riscv_instr;
use crate::emulator::cpu::Cpu;
use crate::instruction::format::format_r::FormatR;
use crate::instruction::format::InstructionFormat;
use crate::instruction::inline_blake2::{
    execute_blake2b_compression, Blake2SequenceBuilder, HASH_STATE_SIZE, MESSAGE_BLOCK_SIZE,
    NEEDED_REGISTERS,
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

/// Load words from memory into slice
fn load_words_from_memory(cpu: &mut Cpu, base_addr: u64, state: &mut [u64]) -> Result<(), String> {
    for (i, word) in state.iter_mut().enumerate() {
        *word = cpu
            .mmu
            .load_doubleword(base_addr.wrapping_add((i * 8) as u64))
            .map_err(|e| {
                format!("BLAKE2: Failed to load at offset {}: {:?}", i * 8, e)
            })?
            .0;
    }
    Ok(())
}

/// Store words to memory from slice
fn store_words_to_memory(cpu: &mut Cpu, base_addr: u64, values: &[u64]) -> Result<(), String> {
    for (i, &value) in values.iter().enumerate() {
        cpu.mmu
            .store_doubleword(base_addr.wrapping_add((i * 8) as u64), value)
            .map_err(|e| {
                format!("BLAKE2: Failed to store at offset {}: {:?}", i * 8, e)
            })?;
    }
    Ok(())
}

impl BLAKE2 {
    /// Fast path emulation using native Rust Blake2b implementation.
    fn exec(&self, cpu: &mut Cpu, _ram_access: &mut <BLAKE2 as RISCVInstruction>::RAMAccess) {
        let state_addr = cpu.x[self.operands.rs1] as u64;
        let block_addr = cpu.x[self.operands.rs2] as u64;

        // Read 8-word hash state
        let mut state = [0u64; HASH_STATE_SIZE];
        load_words_from_memory(cpu, state_addr, &mut state).expect("Failed to load hash state");

        // Read 16-word message block
        let mut message_words = [0u64; MESSAGE_BLOCK_SIZE + 2];
        load_words_from_memory(cpu, block_addr, &mut message_words)
            .expect("Failed to load message block");

        // Load counter value after message block
        message_words[16] = {
            let mut buffer = [0];
            load_words_from_memory(
                cpu,
                block_addr.wrapping_add(MESSAGE_BLOCK_SIZE as u64 * 8),
                &mut buffer,
            )
            .expect("Failed to load counter");
            buffer[0]
        };

        // Load final block flag after counter
        message_words[17] = {
            let mut buffer = [0];
            load_words_from_memory(
                cpu,
                block_addr.wrapping_add(MESSAGE_BLOCK_SIZE as u64 * 8 + 8),
                &mut buffer,
            )
            .expect("Failed to load final flag");
            buffer[0]
        };

        // Execute Blake2b compression
        execute_blake2b_compression(&mut state, &message_words);

        // Write result back to memory
        store_words_to_memory(cpu, state_addr, &state).expect("Failed to store result");
    }
}

impl VirtualInstructionSequence for BLAKE2 {
    fn virtual_sequence(&self) -> Vec<RV32IMInstruction> {
        let vr: [usize; NEEDED_REGISTERS] =
            core::array::from_fn(|i| virtual_register_index(i as u64) as usize);

        Blake2SequenceBuilder::new(self.address, vr, self.operands.rs1, self.operands.rs2).build()
    }
}

impl RISCVTrace for BLAKE2 {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence();

        let mut trace = trace;
        for instr in virtual_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emulator::{cpu::Cpu, default_terminal::DefaultTerminal, mmu::DRAM_BASE};
    use crate::instruction::format::format_r::FormatR;

    const TEST_MEMORY_CAPACITY: u64 = 1024 * 1024; // 1MB

    // Test constants from RFC 7693 Appendix A (Blake2b with "abc")
    const INITIAL_STATE: [u64; HASH_STATE_SIZE] = [
        0x6a09e667f3bcc908,
        0xbb67ae8584caa73b,
        0x3c6ef372fe94f82b,
        0xa54ff53a5f1d36f1,
        0x510e527fade682d1,
        0x9b05688c2b3e6c1f,
        0x1f83d9abfb41bd6b,
        0x5be0cd19137e2179,
    ];

    const EXPECTED_STATE: [u64; HASH_STATE_SIZE] = [
        0x0D4D1C983FA580BAu64, // BA 80 A5 3F 98 1C 4D 0D (little-endian)
        0xE9F6129FB697276Au64, // 6A 27 97 B6 9F 12 F6 E9
        0xB7C45A68142F214Cu64, // 4C 21 2F 14 68 5A C4 B7
        0xD1A2FFDB6FBB124Bu64, // 4B 12 BB 6F DB FF A2 D1
        0x2D79AB2A39C5877Du64, // 7D 87 C5 39 2A AB 79 2D
        0x95CC3345DED552C2u64, // C2 52 D5 DE 45 33 CC 95
        0x5A92F1DBA88AD318u64, // 18 D3 8A A8 DB F1 92 5A
        0x239900D4ED8623B9u64, // B9 23 86 ED D4 00 99 23
    ];

    fn get_pre_post_states() -> ([u64; HASH_STATE_SIZE], [u64; HASH_STATE_SIZE]) {
        (INITIAL_STATE, EXPECTED_STATE)
    }

    /// Test macro to reduce repetitive setup and verification
    macro_rules! test_blake2 {
        ($test_name:ident, $exec_block:expr) => {
            #[test]
            fn $test_name() {
                let (mut initial_state, expected_state) = get_pre_post_states();
                // Apply Blake2b parameter block: h[0] ^= 0x01010000 ^ (kk << 8) ^ nn
                initial_state[0] ^= 0x01010000 ^ (0u64 << 8) ^ 64u64;

                // Message block with "abc" in little-endian
                let mut message_block = [0u64; MESSAGE_BLOCK_SIZE];
                message_block[0] = 0x0000000000636261u64; // "abc"

                let (counter, is_final) = (3u64, true);

                let instruction = BLAKE2 {
                    address: 0,
                    operands: FormatR {
                        rs1: 10, // Points to state
                        rs2: 11, // Points to message block + counter + final flag
                        rd: 0,
                    },
                    virtual_sequence_remaining: None,
                };

                // Set up CPU
                let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
                cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
                let state_addr = DRAM_BASE;
                let message_addr = DRAM_BASE + 1024; // Separate address for message block
                cpu.x[10] = state_addr as i64; // rs1 points to state
                cpu.x[11] = message_addr as i64; // rs2 points to message block

                // Store initial state (8 words) at rs1
                store_words_to_memory(&mut cpu, state_addr, &initial_state)
                    .expect("Failed to store initial state");
                // Store message block (16 words) at rs2
                store_words_to_memory(&mut cpu, message_addr, &message_block)
                    .expect("Failed to store message block");
                // Store counter after message block
                store_words_to_memory(&mut cpu, message_addr + 128, &[counter])
                    .expect("Failed to store counter");
                // Store final flag after counter
                store_words_to_memory(
                    &mut cpu,
                    message_addr + 136,
                    &[if is_final { 1 } else { 0 }],
                )
                .expect("Failed to store final flag");

                // Execute the instruction
                $exec_block(&instruction, &mut cpu);

                // Verify results (Blake2b compression outputs 8 words)
                let mut result = [0u64; HASH_STATE_SIZE];
                for i in 0..HASH_STATE_SIZE {
                    let addr = state_addr + (i * 8) as u64;
                    result[i] = cpu.mmu.load_doubleword(addr).unwrap().0;
                    assert_eq!(
                        result[i], expected_state[i],
                        "Mismatch at word {}: got {:#x}, expected {:#x}",
                        i, result[i], expected_state[i]
                    );
                }
            }
        };
    }

    test_blake2!(
        test_exec_correctness,
        |instruction: &BLAKE2, cpu: &mut Cpu| {
            instruction.exec(cpu, &mut ());
        }
    );

    test_blake2!(
        test_trace_correctness,
        |instruction: &BLAKE2, cpu: &mut Cpu| {
            instruction.trace(cpu, None);
        }
    );
}
