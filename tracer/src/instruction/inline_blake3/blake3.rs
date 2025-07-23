use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

use crate::declare_riscv_instr;
use crate::emulator::cpu::Cpu;
use crate::instruction::format::format_r::FormatR;
use crate::instruction::format::InstructionFormat;
use crate::instruction::inline_blake3::{
    execute_blake3_compression, Blake3SequenceBuilder, CHAINING_VALUE_SIZE, MESSAGE_BLOCK_SIZE, NEEDED_REGISTERS
};
use crate::instruction::{
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction, VirtualInstructionSequence,
};

declare_riscv_instr!(
    name   = BLAKE3,
    mask   = 0xfe00707f,  // Mask for funct7 + funct3 + opcode
    match  = 0x0600000b,  // funct7=0x03, funct3=0x0, opcode=0x0B (custom-0)
    format = FormatR,
    ram    = ()
);

fn load_words_from_memory(cpu: &mut Cpu, base_addr: u64, state: &mut [u32]) {
    for (i, word) in state.iter_mut().enumerate() {
        *word = cpu
            .mmu
            .load_word(base_addr.wrapping_add((i * 4) as u64))
            .expect("BLAKE3: Failed to load from memory")
            .0;
    }
}

fn store_words_to_memory(cpu: &mut Cpu, base_addr: u64, values: &[u32]) {
    for (i, &value) in values.iter().enumerate() {
        cpu.mmu
            .store_word(base_addr.wrapping_add((i * 4) as u64), value)
            .unwrap();
    }
}

impl BLAKE3 {
    // This is the "fast path" for emulation without tracing.
    // It performs the Blake3 compression using a native Rust implementation.
    fn exec(&self, cpu: &mut Cpu, _ram_access: &mut <BLAKE3 as RISCVInstruction>::RAMAccess) {
        // 1. Read the 8-word (32-byte) hash state from memory pointed to by rs1.
        let mut chaining_value = [0u32; CHAINING_VALUE_SIZE];
        let state_addr = cpu.x[self.operands.rs1] as u64;
        load_words_from_memory(cpu, state_addr, &mut chaining_value);

        // 2. Read the 16-word (64-byte) message block from memory pointed to by rs2.
        let mut message_words = [0u32; MESSAGE_BLOCK_SIZE];
        let block_addr = cpu.x[self.operands.rs2] as u64;
        load_words_from_memory(cpu, block_addr, &mut message_words);

        // 3. Load counter values (t0, t1) from memory at offset 64 bytes after message block
        let mut counter = [0u32; 2];
        load_words_from_memory(cpu, block_addr.wrapping_add(MESSAGE_BLOCK_SIZE as u64 * 4), &mut counter);

        // 4. Load input bytes
        let mut input_bytes = [0u32; 1];
        load_words_from_memory(cpu, block_addr.wrapping_add(MESSAGE_BLOCK_SIZE as u64 * 4 + 8), &mut input_bytes);

        // 4. Load input bytes
        let mut flags = [0u32; 1];
        load_words_from_memory(cpu, block_addr.wrapping_add(MESSAGE_BLOCK_SIZE as u64 * 4 + 12), &mut flags);

        // 5. Execute Blake3 compression function with all parameters
        let output = execute_blake3_compression(&chaining_value, &message_words, &counter, input_bytes[0], flags[0]);

        // 6. Write the compressed state back to memory.
        store_words_to_memory(cpu, state_addr, &output);
    }
}

impl VirtualInstructionSequence for BLAKE3 {
    fn virtual_sequence(&self) -> Vec<RV32IMInstruction> {
        let vr: [usize; NEEDED_REGISTERS] =
            core::array::from_fn(|i| virtual_register_index(i as u64) as usize);

        Blake3SequenceBuilder::new(self.address, vr, self.operands.rs1, self.operands.rs2).build()
    }
}

impl RISCVTrace for BLAKE3 {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence();

        let mut trace = trace;
        for instr in virtual_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::emulator::{cpu::Cpu, default_terminal::DefaultTerminal, mmu::DRAM_BASE};
    use crate::instruction::format::format_r::FormatR;
    use crate::instruction::inline_blake3::BLAKE3_IV;

    const TEST_MEMORY_CAPACITY: u64 = 1024 * 1024; // 1MB

    pub const BLOCK_WORDS: [u32; 16] = [0u32, 1u32, 2u32, 3u32, 4u32, 5u32, 6u32, 7u32, 8u32, 9u32, 10u32, 11u32, 12u32, 13u32, 14u32, 15u32];
    pub const COUNTER: [u32; 2] = [0, 0];
    pub const BLOCK_LEN: u32 = 64;
    pub const FLAGS: u32 = 0;
    pub const EXPECTED_RESULTS: [u32; 16] = [0x5f98b37e, 0x26b0af2a, 0xdc58b278, 0x85d56ff6, 0x96f5d384, 0x42c9e776, 0xbeedd1e4, 0xa03faf22, 0x8a4b2d59, 0x1a1c224d, 0x303f2ae7, 0xd36ee60c, 0xfba05dbb, 0xef024714, 0xf597a6be, 0xd849c813];

    #[test]
    fn test_exec_correctness() {
        let instruction = BLAKE3 {
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

        // Store initial state (8 words = 32 bytes) at rs1
        store_words_to_memory(&mut cpu_exec, state_addr, &BLAKE3_IV);
        // Store message block (16 words = 64 bytes) at rs2
        store_words_to_memory(&mut cpu_exec, message_addr, &BLOCK_WORDS);
        // Store counter at rs2 + 64 (4 bytes)
        store_words_to_memory(&mut cpu_exec, message_addr + 64, &COUNTER);

        store_words_to_memory(&mut cpu_exec, message_addr + 72, &[BLOCK_LEN]);
        store_words_to_memory(&mut cpu_exec, message_addr + 76, &[FLAGS]);

        instruction.exec(&mut cpu_exec, &mut ());

        let mut actual_result = [0u32; 16];
        for i in 0..16 {
            actual_result[i] = cpu_exec.mmu.load_word(state_addr + (i * 4) as u64).unwrap().0;
            assert_eq!(actual_result[i], EXPECTED_RESULTS[i]);
        }
    }

    #[test]
    fn test_trace_correctness() {
        let instruction = BLAKE3 {
            address: 0,
            operands: FormatR {
                rs1: 10, // Points to state
                rs2: 11, // Points to message block + counter + final flag
                rd: 0,
            },
            virtual_sequence_remaining: None,
        };
        // Set up the "exec" path CPU
        let mut cpu_trace = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu_trace.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        let state_addr = DRAM_BASE;
        let message_addr = DRAM_BASE + 1024; // Separate address for message block
        cpu_trace.x[10] = state_addr as i64; // rs1 points to state
        cpu_trace.x[11] = message_addr as i64; // rs2 points to message block

        // Store initial state (8 words = 32 bytes) at rs1
        store_words_to_memory(&mut cpu_trace, state_addr, &BLAKE3_IV);
        // Store message block (16 words = 64 bytes) at rs2
        store_words_to_memory(&mut cpu_trace, message_addr, &BLOCK_WORDS);
        // Store counter at rs2 + 64 (4 bytes)
        store_words_to_memory(&mut cpu_trace, message_addr + 64, &COUNTER);

        store_words_to_memory(&mut cpu_trace, message_addr + 72, &[BLOCK_LEN]);
        store_words_to_memory(&mut cpu_trace, message_addr + 76, &[FLAGS]);

        instruction.trace(&mut cpu_trace, None);

        let mut result = [0u32; 16];
        for i in 0..16 {
            result[i] = cpu_trace.mmu.load_word(state_addr + (i * 4) as u64).unwrap().0;
            println!("Index[{:2}] - result = 0x{:08x} ", i, result[i]);
        }

        let mut result = [0u32; 16];
        for i in 0..16 {
            result[i] = cpu_trace.mmu.load_word(state_addr + (i * 4) as u64).unwrap().0;
            assert_eq!(result[i], EXPECTED_RESULTS[i], "Index[{:2}] - result = 0x{:08x}, Expected result = 0x{:08x} ", i, result[i], EXPECTED_RESULTS[i]);
        }
    }
} 