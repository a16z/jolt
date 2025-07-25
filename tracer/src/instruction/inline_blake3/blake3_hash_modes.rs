use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

use crate::declare_riscv_instr;
use crate::emulator::cpu::Cpu;
use crate::instruction::format::format_r::FormatR;
use crate::instruction::format::InstructionFormat;
use crate::instruction::inline_blake3::{
    execute_blake3_compression, Blake3SequenceBuilder, BLAKE3_IV, MESSAGE_BLOCK_SIZE,
    NEEDED_REGISTERS,
};
use crate::instruction::{
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction, VirtualInstructionSequence,
};

// Declare the 4 Blake3 variants with different funct7 values
declare_riscv_instr!(
    name   = BLAKE3_64,
    mask   = 0xfe00707f,  // Mask for funct7 + funct3 + opcode
    match  = 0x0600200b,  // funct7=0x03, funct3=0x2, opcode=0x0B (custom-0)
    format = FormatR,
    ram    = ()
);

declare_riscv_instr!(
    name   = BLAKE3_128,
    mask   = 0xfe00707f,  // Mask for funct7 + funct3 + opcode
    match  = 0x0600300b,  // funct7=0x03, funct3=0x3, opcode=0x0B (custom-0)
    format = FormatR,
    ram    = ()
);

declare_riscv_instr!(
    name   = BLAKE3_192,
    mask   = 0xfe00707f,  // Mask for funct7 + funct3 + opcode
    match  = 0x0600400b,  // funct7=0x03, funct3=0x4, opcode=0x0B (custom-0)
    format = FormatR,
    ram    = ()
);

declare_riscv_instr!(
    name   = BLAKE3_256,
    mask   = 0xfe00707f,  // Mask for funct7 + funct3 + opcode
    match  = 0x0600500b,  // funct7=0x03, funct3=0x5, opcode=0x0B (custom-0)
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

/// Macro to implement the exec method for Blake3 variants with chained compressions
macro_rules! impl_blake3_exec {
    ($struct_name:ident, $num_blocks:expr) => {
        impl $struct_name {
            /// Fast path for emulation without tracing.
            /// Performs Blake3 compression using a native Rust implementation.
            fn exec(
                &self,
                cpu: &mut Cpu,
                _ram_access: &mut <$struct_name as RISCVInstruction>::RAMAccess,
            ) {
                // Memory addresses
                let state_addr = cpu.x[self.operands.rs1] as u64;
                let block_addr = cpu.x[self.operands.rs2] as u64;

                // Initialize chaining value with BLAKE3_IV
                let mut chaining_value = [0u32; 16];
                chaining_value[0..8].copy_from_slice(&BLAKE3_IV);

                // Perform chained compressions
                for i in 0..$num_blocks {
                    // Load message block for this iteration
                    let mut message_words = [0u32; MESSAGE_BLOCK_SIZE];
                    let message_offset = (i * MESSAGE_BLOCK_SIZE * 4) as u64;
                    load_words_from_memory(cpu, block_addr + message_offset, &mut message_words)
                        .expect("Failed to load message block");

                    // Set flags: chunk_start (bit 0), chunk_end (bit 1), root (bit 3)
                    let chunk_start = i == 0;
                    let chunk_end = i == $num_blocks - 1;
                    let flags = (chunk_start as u32)
                        | ((chunk_end as u32) << 1)
                        | ((chunk_end as u32) << 3);

                    // Execute Blake3 compression function
                    execute_blake3_compression(
                        &mut chaining_value,
                        &message_words,
                        &[0, 0], // counter (not used for hashing mode)
                        64,      // block length
                        flags,
                    );
                }

                // Store the final result back to memory
                store_words_to_memory(cpu, state_addr, &chaining_value)
                    .expect("Failed to store result");
            }
        }
    };
}

/// Macro to implement VirtualInstructionSequence for Blake3 variants
macro_rules! impl_virtual_instruction_sequence {
    ($struct_name:ident, $hash_param:expr) => {
        impl VirtualInstructionSequence for $struct_name {
            fn virtual_sequence(&self) -> Vec<RV32IMInstruction> {
                let vr: [usize; NEEDED_REGISTERS] =
                    core::array::from_fn(|i| virtual_register_index(i as u64) as usize);

                Blake3SequenceBuilder::new(
                    self.address,
                    vr,
                    self.operands.rs1,
                    self.operands.rs2,
                    super::BuilderMode::HASH($hash_param),
                )
                .build()
            }
        }
    };
}

/// Macro to implement RISCVTrace for Blake3 variants
macro_rules! impl_riscv_trace {
    ($struct_name:ident) => {
        impl RISCVTrace for $struct_name {
            fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
                let virtual_sequence = self.virtual_sequence();

                let mut trace = trace;
                for instr in virtual_sequence {
                    instr.trace(cpu, trace.as_deref_mut());
                }
            }
        }
    };
}

// Implement all traits for each Blake3 variant
impl_blake3_exec!(BLAKE3_64, 1);
impl_virtual_instruction_sequence!(BLAKE3_64, 1);
impl_riscv_trace!(BLAKE3_64);

impl_blake3_exec!(BLAKE3_128, 2);
impl_virtual_instruction_sequence!(BLAKE3_128, 2);
impl_riscv_trace!(BLAKE3_128);

impl_blake3_exec!(BLAKE3_192, 3);
impl_virtual_instruction_sequence!(BLAKE3_192, 3);
impl_riscv_trace!(BLAKE3_192);

impl_blake3_exec!(BLAKE3_256, 4);
impl_virtual_instruction_sequence!(BLAKE3_256, 4);
impl_riscv_trace!(BLAKE3_256);

#[cfg(test)]
mod compression_tests {
    use super::*;
    use crate::emulator::{cpu::Cpu, default_terminal::DefaultTerminal, mmu::DRAM_BASE};
    use crate::instruction::format::format_r::FormatR;

    const TEST_MEMORY_CAPACITY: u64 = 1024 * 1024; // 1MB

    // 2D array with 4 blocks, each containing 16 words (shown in hex for better visualization)
    #[rustfmt::skip]
    pub const BLOCK_WORDS: [[u32; 16]; 4] = [
        // Block 0: Sequential byte pattern 0x00010203...0x3c3d3e3f
        [0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c, 0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c,
         0x23222120, 0x27262524, 0x2b2a2928, 0x2f2e2d2c, 0x33323130, 0x37363534, 0x3b3a3938, 0x3f3e3d3c],
        // Block 1: Sequential byte pattern 0x40414243...0x7c7d7e7f
        [0x43424140, 0x47464544, 0x4b4a4948, 0x4f4e4d4c, 0x53525150, 0x57565554, 0x5b5a5958, 0x5f5e5d5c,
         0x63626160, 0x67666564, 0x6b6a6968, 0x6f6e6d6c, 0x73727170, 0x77767574, 0x7b7a7978, 0x7f7e7d7c],
        // Block 2: Sequential byte pattern 0x80818283...0xbcbdbebf
        [0x83828180, 0x87868584, 0x8b8a8988, 0x8f8e8d8c, 0x93929190, 0x97969594, 0x9b9a9998, 0x9f9e9d9c,
         0xa3a2a1a0, 0xa7a6a5a4, 0xabaaa9a8, 0xafaeadac, 0xb3b2b1b0, 0xb7b6b5b4, 0xbbbab9b8, 0xbfbebdbc],
        // Block 3: Sequential byte pattern 0xc0c1c2c3...0xfcfdfeff
        [0xc3c2c1c0, 0xc7c6c5c4, 0xcbcac9c8, 0xcfcecdcc, 0xd3d2d1d0, 0xd7d6d5d4, 0xdbdad9d8, 0xdfdedddc,
         0xe3e2e1e0, 0xe7e6e5e4, 0xebeae9e8, 0xefeeedec, 0xf3f2f1f0, 0xf7f6f5f4, 0xfbfaf9f8, 0xfffefdfc],
    ];

    // Expected results for each Blake3 mode (64, 128, 192, 256)
    pub const EXPECTED_RESULTS: [[u32; 16]; 4] = [
        // BLAKE3_64 (1 block)
        [
            0x4171ed4e, 0xd45c4aea, 0x6b6088b7, 0xe2463fd2, 0xac9caf12, 0x7ddcaceb, 0xc76d4c1f,
            0x981b51f2, 0x6cc59cfc, 0xe3ff31b8, 0xe1e7a83e, 0xb209dfd1, 0x6727fd6e, 0xaa660067,
            0xb123d082, 0x1babe8df,
        ],
        // BLAKE3_128 (2 blocks)
        [
            0x5577ef1, 0x7865b264, 0xf4b73bc3, 0x39f54346, 0xdf054b62, 0x1fc8761a, 0x48d5ac30,
            0xef454bc4, 0xa0ab9fa6, 0x9c7f4291, 0x87aa4c5c, 0x2878a03a, 0xc5191f65, 0xc485ad5b,
            0xb168137d, 0x9ed96f1c,
        ],
        // BLAKE3_192 (3 blocks)
        [
            0x235dbc4a, 0xcc4afb28, 0x4bff9a54, 0xbf07d87, 0x97462da5, 0x6b9d7457, 0x58d4338c,
            0xfffcb870, 0xdd896cf8, 0xa5f40732, 0xe842d97b, 0x4caa3db1, 0x420a25b, 0x2cff4eb8,
            0xfd58405c, 0x9dcac30b,
        ],
        // BLAKE3_256 (4 blocks)
        [
            0xa45b494a, 0x8e746124, 0xd6da8fca, 0xaa76f918, 0x90c26c72, 0xb4fce93d, 0x86a73507,
            0x6b191cac, 0x94fc88e3, 0xb022df83, 0xe7d57d4a, 0xcb558b67, 0x70fc9994, 0x7961680b,
            0x84fb2342, 0x8d4015ab,
        ],
    ];

    /// Macro to reduce repetitive test setup and verification code
    macro_rules! test_blake3_variant {
        ($test_name:ident, $struct_name:ident, $num_blocks:expr, $exec_block:expr) => {
            #[test]
            fn $test_name() {
                let instruction = $struct_name {
                    address: 0,
                    operands: FormatR {
                        rs1: 10, // Points to state
                        rs2: 11, // Points to message block
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

                // Store the required number of message blocks
                for block_idx in 0..$num_blocks {
                    let block_offset = (block_idx * 16 * 4) as u64; // 16 words * 4 bytes per word
                    store_words_to_memory(
                        &mut cpu,
                        message_addr + block_offset,
                        &BLOCK_WORDS[block_idx],
                    )
                    .expect("Failed to store message block");
                }

                // Execute the instruction
                $exec_block(&instruction, &mut cpu);

                // Verify that execution completed without errors
                // (Full result verification would require known test vectors for each variant)
                let mut results = [0u32; 16];
                for i in 0..16 {
                    results[i] = cpu.mmu.load_word(state_addr + (i * 4) as u64).unwrap().0;
                }

                // Verify results
                let mut results = [0u32; 16];
                for i in 0..16 {
                    results[i] = cpu.mmu.load_word(state_addr + (i * 4) as u64).unwrap().0;
                    assert_eq!(
                        results[i],
                        EXPECTED_RESULTS[$num_blocks - 1][i],
                        "Mismatch at word {}: got {:#x}, expected {:#x}",
                        i,
                        results[i],
                        EXPECTED_RESULTS[$num_blocks - 1][i]
                    );
                }
            }
        };
    }

    // Test exec methods
    test_blake3_variant!(
        test_blake3_64_exec,
        BLAKE3_64,
        1,
        |instruction: &BLAKE3_64, cpu: &mut Cpu| {
            instruction.exec(cpu, &mut ());
        }
    );

    test_blake3_variant!(
        test_blake3_128_exec,
        BLAKE3_128,
        2,
        |instruction: &BLAKE3_128, cpu: &mut Cpu| {
            instruction.exec(cpu, &mut ());
        }
    );

    test_blake3_variant!(
        test_blake3_192_exec,
        BLAKE3_192,
        3,
        |instruction: &BLAKE3_192, cpu: &mut Cpu| {
            instruction.exec(cpu, &mut ());
        }
    );

    test_blake3_variant!(
        test_blake3_256_exec,
        BLAKE3_256,
        4,
        |instruction: &BLAKE3_256, cpu: &mut Cpu| {
            instruction.exec(cpu, &mut ());
        }
    );

    // Test trace methods
    test_blake3_variant!(
        test_blake3_64_trace,
        BLAKE3_64,
        1,
        |instruction: &BLAKE3_64, cpu: &mut Cpu| {
            instruction.trace(cpu, None);
        }
    );

    test_blake3_variant!(
        test_blake3_128_trace,
        BLAKE3_128,
        2,
        |instruction: &BLAKE3_128, cpu: &mut Cpu| {
            instruction.trace(cpu, None);
        }
    );

    test_blake3_variant!(
        test_blake3_192_trace,
        BLAKE3_192,
        3,
        |instruction: &BLAKE3_192, cpu: &mut Cpu| {
            instruction.trace(cpu, None);
        }
    );

    test_blake3_variant!(
        test_blake3_256_trace,
        BLAKE3_256,
        4,
        |instruction: &BLAKE3_256, cpu: &mut Cpu| {
            instruction.trace(cpu, None);
        }
    );
}
