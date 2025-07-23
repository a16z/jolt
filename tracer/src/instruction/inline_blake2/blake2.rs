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

fn load_words_from_memory(cpu: &mut Cpu, base_addr: u64, state: &mut [u64]) {
    for (i, word) in state.iter_mut().enumerate() {
        *word = cpu
            .mmu
            .load_doubleword(base_addr.wrapping_add((i * 8) as u64))
            .expect("BLAKE2: Failed to load from memory")
            .0;
    }
}

fn store_words_to_memory(cpu: &mut Cpu, base_addr: u64, values: &[u64]) {
    for (i, &value) in values.iter().enumerate() {
        cpu.mmu
            .store_doubleword(base_addr.wrapping_add((i * 8) as u64), value)
            .unwrap();
    }
}

impl BLAKE2 {
    // This is the "fast path" for emulation without tracing.
    // It performs the Blake2b compression using a native Rust implementation.
    fn exec(&self, cpu: &mut Cpu, _ram_access: &mut <BLAKE2 as RISCVInstruction>::RAMAccess) {
        // 1. Read the 8-word (64-byte) hash state from memory pointed to by rs1.
        let mut state = [0u64; HASH_STATE_SIZE];
        let state_addr = cpu.x[self.operands.rs1] as u64;
        load_words_from_memory(cpu, state_addr, &mut state);

        // 2. Read the 16-word (128-byte) message block from memory pointed to by rs2.
        let mut message_words = [0u64; MESSAGE_BLOCK_SIZE];
        let block_addr = cpu.x[self.operands.rs2] as u64;
        load_words_from_memory(cpu, block_addr, &mut message_words);

        // 3. Load counter value (t) from memory at offset 128 bytes after message block
        let counter = {
            let mut buffer = [0];
            load_words_from_memory(
                cpu,
                block_addr.wrapping_add(MESSAGE_BLOCK_SIZE as u64 * 8),
                &mut buffer,
            );
            buffer[0]
        };

        // 4. Load final block flag (is_final) from memory at offset 136 bytes after message block
        let is_final = {
            let mut buffer = [0];
            load_words_from_memory(
                cpu,
                block_addr.wrapping_add(MESSAGE_BLOCK_SIZE as u64 * 8 + 8),
                &mut buffer,
            );
            buffer[0] != 0
        };

        // 5. Execute Blake2b compression function with all parameters
        execute_blake2b_compression(&mut state, &message_words, counter, is_final);

        // 6. Write the compressed state back to memory.
        store_words_to_memory(cpu, state_addr, &state);
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
    fn get_pre_post_states() -> ([u64; 8], [u64; 8]) {
        // Values are from RFC 7693 -> Appendix A
        let pre_state = [
            0x6a09e667f3bcc908,
            0xbb67ae8584caa73b,
            0x3c6ef372fe94f82b,
            0xa54ff53a5f1d36f1,
            0x510e527fade682d1,
            0x9b05688c2b3e6c1f,
            0x1f83d9abfb41bd6b,
            0x5be0cd19137e2179,
        ];
        let post_state = [
            0x0D4D1C983FA580BAu64, // BA 80 A5 3F 98 1C 4D 0D (little-endian)
            0xE9F6129FB697276Au64, // 6A 27 97 B6 9F 12 F6 E9
            0xB7C45A68142F214Cu64, // 4C 21 2F 14 68 5A C4 B7
            0xD1A2FFDB6FBB124Bu64, // 4B 12 BB 6F DB FF A2 D1
            0x2D79AB2A39C5877Du64, // 7D 87 C5 39 2A AB 79 2D
            0x95CC3345DED552C2u64, // C2 52 D5 DE 45 33 CC 95
            0x5A92F1DBA88AD318u64, // 18 D3 8A A8 DB F1 92 5A
            0x239900D4ED8623B9u64, // B9 23 86 ED D4 00 99 23
        ];
        (pre_state, post_state)
    }

    #[test]
    fn test_exec_correctness() {
        let (mut initial_state, expected_state) = get_pre_post_states();
        initial_state[0] ^= 0x01010000 ^ (0u64 << 8) ^ 64u64;
        let message_block = {
            let mut msg = [0u64; 16];
            msg[0] = 0x0000000000636261u64; // "abc" in little-endian
            msg
        };
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
        // Set up the "exec" path CPU
        let mut cpu_exec = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu_exec.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        let state_addr = DRAM_BASE;
        let message_addr = DRAM_BASE + 1024; // Separate address for message block
        cpu_exec.x[10] = state_addr as i64; // rs1 points to state
        cpu_exec.x[11] = message_addr as i64; // rs2 points to message block

        // Store initial state (8 words = 64 bytes) at rs1
        store_words_to_memory(&mut cpu_exec, state_addr, &initial_state);
        // Store message block (16 words = 128 bytes) at rs2
        store_words_to_memory(&mut cpu_exec, message_addr, &message_block);
        // Store counter at rs2 + 128 (8 bytes)
        store_words_to_memory(&mut cpu_exec, message_addr + 128, &[counter]);
        // Store final flag at rs2 + 136 (8 bytes)
        store_words_to_memory(
            &mut cpu_exec,
            message_addr + 136,
            &[if is_final { 1 } else { 0 }],
        );

        instruction.exec(&mut cpu_exec, &mut ());

        let mut actual_result = [0u64; 8];
        for i in 0..8 {
            let addr = state_addr + (i * 8) as u64;
            actual_result[i] = cpu_exec.mmu.load_doubleword(addr).unwrap().0;
            assert_eq!(actual_result[i], expected_state[i]);
        }
    }

    #[test]
    fn test_trace_correctness() {
        let (mut initial_state, expected_state) = get_pre_post_states();
        initial_state[0] ^= 0x01010000 ^ (0u64 << 8) ^ 64u64;
        let message_block = {
            let mut msg = [0u64; 16];
            msg[0] = 0x0000000000636261u64; // "abc" in little-endian
            msg
        };
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
        // Set up the "trace" path CPU
        let mut cpu_trace = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu_trace.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        let state_addr = DRAM_BASE;
        let message_addr = DRAM_BASE + 1024; // Separate address for message block
        cpu_trace.x[10] = state_addr as i64; // rs1 points to state
        cpu_trace.x[11] = message_addr as i64; // rs2 points to message block

        // Store initial state (8 words = 64 bytes) at rs1
        store_words_to_memory(&mut cpu_trace, state_addr, &initial_state);
        // Store message block (16 words = 128 bytes) at rs2
        store_words_to_memory(&mut cpu_trace, message_addr, &message_block);
        // Store counter at rs2 + 128 (8 bytes)
        store_words_to_memory(&mut cpu_trace, message_addr + 128, &[counter]);
        // Store final flag at rs2 + 136 (8 bytes)
        store_words_to_memory(
            &mut cpu_trace,
            message_addr + 136,
            &[if is_final { 1 } else { 0 }],
        );

        instruction.trace(&mut cpu_trace, None);

        // Get expected working state from reference implementation
        use crate::instruction::inline_blake2::execute_blake2b_compression;
        execute_blake2b_compression(&mut initial_state, &message_block, counter, is_final);

        // Check final hash state
        let mut actual_result = [0u64; 8];
        for i in 0..8 {
            let addr = state_addr + (i * 8) as u64;
            actual_result[i] = cpu_trace.mmu.load_doubleword(addr).unwrap().0;
            assert_eq!(actual_result[i], expected_state[i], "value: {}", i);
        }
    }
}
