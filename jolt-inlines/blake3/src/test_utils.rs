// Test utilities for BLAKE3 instruction tests
//
// This module contains BLAKE3-specific setup code, utilities, and helper functions
// to reduce code duplication in the test suite. It relies on the generic
// `CpuTestHarness` for the underlying emulator setup.

use crate::trace_generator::NEEDED_REGISTERS;
use jolt_inlines_common::constants::{blake3, INLINE_OPCODE};
use tracer::emulator::mmu::DRAM_BASE;
use tracer::instruction::format::format_inline::FormatInline;
use tracer::instruction::inline::INLINE;
use tracer::utils::test_harness::CpuTestHarness;

// BLAKE3 constants
pub const HASH_STATE_SIZE: usize = 8; // 8 u32 words for chaining value
pub const MESSAGE_BLOCK_SIZE: usize = 16; // 16 u32 words
pub const WORKING_STATE_SIZE: usize = 16; // 16 u32 words for working state

/// BLAKE3-specific CPU test harness.
/// Wrapper around `CpuTestHarness` that offers convenient BLAKE3 helpers.
pub struct Blake3CpuHarness {
    pub harness: CpuTestHarness,
    pub vr: [u8; NEEDED_REGISTERS as usize],
}

impl Blake3CpuHarness {
    /// Memory layout constants (all using 32-bit words)
    const CHAINING_VALUE_ADDR: u64 = DRAM_BASE;
    const MESSAGE_ADDR: u64 = DRAM_BASE + (HASH_STATE_SIZE * 4) as u64; // 8 * 4 bytes
    const COUNTER_ADDR: u64 = Self::MESSAGE_ADDR + (MESSAGE_BLOCK_SIZE * 4) as u64; // After message block
    const BLOCK_LEN_ADDR: u64 = Self::COUNTER_ADDR + 8; // Counter is 2 u32s = 8 bytes
    const FLAGS_ADDR: u64 = Self::BLOCK_LEN_ADDR + 4; // Block len is 1 u32 = 4 bytes

    /// Register assignments for the instruction
    pub const RS1: u8 = 10; // Points to chaining value
    pub const RS2: u8 = 11; // Points to message block + counter + block_len + flags

    /// Create a new harness with initialized memory.
    pub fn new() -> Self {
        // Allocate virtual registers
        let guards: Vec<_> = (0..NEEDED_REGISTERS)
            .map(|_| tracer::utils::virtual_registers::allocate_virtual_register_for_inline())
            .collect();
        let vr: [u8; NEEDED_REGISTERS as usize] = core::array::from_fn(|i| *guards[i]);

        Self {
            harness: CpuTestHarness::new(),
            vr,
        }
    }

    /// Load state and message data into DRAM and set up registers
    pub fn load_blake3_data(
        &mut self,
        chaining_value: &[u32; HASH_STATE_SIZE],
        message: &[u32; MESSAGE_BLOCK_SIZE],
        counter: &[u32; 2],
        block_len: u32,
        flags: u32,
    ) {
        // Set up memory pointers in registers
        self.harness.cpu.x[Self::RS1 as usize] = Self::CHAINING_VALUE_ADDR as i64;
        self.harness.cpu.x[Self::RS2 as usize] = Self::MESSAGE_ADDR as i64;

        // Load chaining value into memory (as u32 words)
        for (i, &word) in chaining_value.iter().enumerate() {
            self.harness
                .cpu
                .mmu
                .store_word(Self::CHAINING_VALUE_ADDR.wrapping_add((i * 4) as u64), word)
                .expect("BLAKE3: Failed to store chaining value to memory");
        }

        // Load message block into memory (as u32 words)
        for (i, &word) in message.iter().enumerate() {
            self.harness
                .cpu
                .mmu
                .store_word(Self::MESSAGE_ADDR.wrapping_add((i * 4) as u64), word)
                .expect("BLAKE3: Failed to store message to memory");
        }

        // Load counter (2 u32 words)
        self.harness
            .cpu
            .mmu
            .store_word(Self::COUNTER_ADDR, counter[0])
            .expect("BLAKE3: Failed to store counter[0] to memory");
        self.harness
            .cpu
            .mmu
            .store_word(Self::COUNTER_ADDR.wrapping_add(4), counter[1])
            .expect("BLAKE3: Failed to store counter[1] to memory");

        // Load block length
        self.harness
            .cpu
            .mmu
            .store_word(Self::BLOCK_LEN_ADDR, block_len)
            .expect("BLAKE3: Failed to store block_len to memory");

        // Load flags
        self.harness
            .cpu
            .mmu
            .store_word(Self::FLAGS_ADDR, flags)
            .expect("BLAKE3: Failed to store flags to memory");
    }

    /// Read the hash state (chaining value) from DRAM
    pub fn read_chaining_value(&mut self) -> [u32; HASH_STATE_SIZE] {
        let mut chaining_value = [0u32; HASH_STATE_SIZE];
        for (i, word) in chaining_value.iter_mut().enumerate() {
            *word = self
                .harness
                .cpu
                .mmu
                .load_word(Self::CHAINING_VALUE_ADDR.wrapping_add((i * 4) as u64))
                .expect("BLAKE3: Failed to load chaining value from memory")
                .0;
        }
        chaining_value
    }

    /// Read the working state from DRAM
    pub fn read_working_state(&mut self) -> [u32; WORKING_STATE_SIZE] {
        let mut state = [0u32; WORKING_STATE_SIZE];
        for (i, word) in state.iter_mut().enumerate() {
            *word = self
                .harness
                .cpu
                .mmu
                .load_word(Self::CHAINING_VALUE_ADDR.wrapping_add((i * 4) as u64))
                .expect("BLAKE3: Failed to load working state from memory")
                .0;
        }
        state
    }

    pub fn instruction() -> INLINE {
        INLINE {
            address: 0,
            operands: FormatInline {
                rs1: Self::RS1,
                rs2: Self::RS2,
                rs3: 0,
            },
            opcode: INLINE_OPCODE,
            funct3: blake3::FUNCT3,
            funct7: blake3::FUNCT7,
            inline_sequence_remaining: None,
            is_compressed: false,
        }
    }
}

impl Default for Blake3CpuHarness {
    fn default() -> Self {
        Self::new()
    }
}
