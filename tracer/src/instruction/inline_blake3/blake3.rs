use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

use crate::declare_riscv_instr;
use crate::emulator::cpu::Cpu;
use crate::instruction::format::format_r::FormatR;
use crate::instruction::format::InstructionFormat;
use crate::instruction::inline_blake3::{
    execute_blake3_compression, Blake3SequenceBuilder, CHAINING_VALUE_SIZE, MESSAGE_BLOCK_SIZE,
    NEEDED_REGISTERS,
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

/// Load words from memory into the provided slice
/// Returns an error if any memory access fails
fn load_words_from_memory(cpu: &mut Cpu, base_addr: u64, state: &mut [u32]) -> Result<(), String> {
    for (i, word) in state.iter_mut().enumerate() {
        *word = cpu
            .mmu
            .load_word(base_addr.wrapping_add((i * 4) as u64))
            .map_err(|e| {
                format!(
                    "BLAKE3: Failed to load from memory at offset {}: {:?}",
                    i * 4,
                    e
                )
            })?
            .0;
    }
    Ok(())
}

/// Store words to memory from the provided slice
/// Returns an error if any memory access fails
fn store_words_to_memory(cpu: &mut Cpu, base_addr: u64, values: &[u32]) -> Result<(), String> {
    for (i, &value) in values.iter().enumerate() {
        cpu.mmu
            .store_word(base_addr.wrapping_add((i * 4) as u64), value)
            .map_err(|e| {
                format!(
                    "BLAKE3: Failed to store to memory at offset {}: {:?}",
                    i * 4,
                    e
                )
            })?;
    }
    Ok(())
}

impl BLAKE3 {
    /// Fast path for emulation without tracing.
    /// Performs Blake3 compression using a native Rust implementation.
    fn exec(&self, cpu: &mut Cpu, _ram_access: &mut <BLAKE3 as RISCVInstruction>::RAMAccess) {
        // Memory addresses
        let state_addr = cpu.x[self.operands.rs1] as u64;
        let block_addr = cpu.x[self.operands.rs2] as u64;

        // 1. Read the 8-word chaining value from memory
        let mut chaining_value = [0u32; CHAINING_VALUE_SIZE * 2];
        load_words_from_memory(cpu, state_addr, &mut chaining_value)
            .expect("Failed to load chaining value");

        // 2. Read the 16-word message block from memory
        let mut message_words = [0u32; MESSAGE_BLOCK_SIZE];
        load_words_from_memory(cpu, block_addr, &mut message_words)
            .expect("Failed to load message block");

        // 3. Load counter values from memory (2 words after message block)
        let mut counter = [0u32; 2];
        load_words_from_memory(
            cpu,
            block_addr.wrapping_add(MESSAGE_BLOCK_SIZE as u64 * 4),
            &mut counter,
        )
        .expect("Failed to load counter");

        // 4. Load input bytes length (1 word after counter)
        let mut input_bytes = [0u32; 1];
        load_words_from_memory(
            cpu,
            block_addr.wrapping_add(MESSAGE_BLOCK_SIZE as u64 * 4 + 8),
            &mut input_bytes,
        )
        .expect("Failed to load input bytes length");

        // 5. Load flags (1 word after input bytes length)
        let mut flags = [0u32; 1];
        load_words_from_memory(
            cpu,
            block_addr.wrapping_add(MESSAGE_BLOCK_SIZE as u64 * 4 + 12),
            &mut flags,
        )
        .expect("Failed to load flags");

        // 6. Execute Blake3 compression function
        execute_blake3_compression(
            &mut chaining_value,
            &message_words,
            &counter,
            input_bytes[0],
            flags[0],
        );

        // 7. Write the result back to memory
        // Blake3 compression returns 16 words, but we only store the first 8 as chaining value
        store_words_to_memory(cpu, state_addr, &chaining_value).expect("Failed to store result");
    }
}

impl VirtualInstructionSequence for BLAKE3 {
    fn virtual_sequence(&self) -> Vec<RV32IMInstruction> {
        let vr: [usize; NEEDED_REGISTERS] =
            core::array::from_fn(|i| virtual_register_index(i as u64) as usize);

        Blake3SequenceBuilder::new(
            self.address,
            vr,
            self.operands.rs1,
            self.operands.rs2,
            super::BuilderMode::COMPRESSION,
        )
        .build()
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
mod compression_tests {
    use super::*;
    use crate::emulator::{cpu::Cpu, default_terminal::DefaultTerminal, mmu::DRAM_BASE};
    use crate::instruction::format::format_r::FormatR;
    use crate::instruction::inline_blake3::BLAKE3_IV;

    const TEST_MEMORY_CAPACITY: u64 = 1024 * 1024; // 1MB

    pub const BLOCK_WORDS: [u32; 16] = [
        50462976, 117835012, 185207048, 252579084, 319951120, 387323156, 454695192, 522067228,
        589439264, 656811300, 724183336, 791555372, 858927408, 926299444, 993671480, 1061043516,
    ];
    pub const COUNTER: [u32; 2] = [0, 0];
    pub const BLOCK_LEN: u32 = 64;
    pub const FLAGS: u32 = 1u32 | 2u32 | 8u32;

    pub const EXPECTED_RESULTS: [u32; 16] = [
        0x4171ed4e, 0xd45c4aea, 0x6b6088b7, 0xe2463fd2, 0xac9caf12, 0x7ddcaceb, 0xc76d4c1f,
        0x981b51f2, 0x6cc59cfc, 0xe3ff31b8, 0xe1e7a83e, 0xb209dfd1, 0x6727fd6e, 0xaa660067,
        0xb123d082, 0x1babe8df,
    ];

    /// Macro to reduce repetitive test setup and verification code
    macro_rules! test_blake3 {
        ($test_name:ident, $exec_block:expr) => {
            #[test]
            fn $test_name() {
                let instruction = BLAKE3 {
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
                store_words_to_memory(&mut cpu, state_addr, &BLAKE3_IV)
                    .expect("Failed to store initial state");
                // Store message block (16 words) at rs2
                store_words_to_memory(&mut cpu, message_addr, &BLOCK_WORDS)
                    .expect("Failed to store message block");
                // Store counter (2 words) after message block
                store_words_to_memory(&mut cpu, message_addr + 64, &COUNTER)
                    .expect("Failed to store counter");
                // Store input bytes length after counter
                store_words_to_memory(&mut cpu, message_addr + 72, &[BLOCK_LEN])
                    .expect("Failed to store input bytes length");
                // Store flags after input bytes length
                store_words_to_memory(&mut cpu, message_addr + 76, &[FLAGS])
                    .expect("Failed to store flags");

                // Execute the instruction
                $exec_block(&instruction, &mut cpu);

                // Verify results
                let mut results = [0u32; 16];
                for i in 0..16 {
                    results[i] = cpu.mmu.load_word(state_addr + (i * 4) as u64).unwrap().0;
                    assert_eq!(
                        results[i], EXPECTED_RESULTS[i],
                        "Mismatch at word {}: got {:#x}, expected {:#x}",
                        i, results[i], EXPECTED_RESULTS[i]
                    );
                }
            }
        };
    }

    test_blake3!(
        test_exec_correctness,
        |instruction: &BLAKE3, cpu: &mut Cpu| {
            instruction.exec(cpu, &mut ());
        }
    );

    test_blake3!(
        test_trace_correctness,
        |instruction: &BLAKE3, cpu: &mut Cpu| {
            instruction.trace(cpu, None);
        }
    );
}
