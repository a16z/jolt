// Test utilities for BLAKE2 instruction tests
//
// This module contains BLAKE2-specific setup code, utilities, and helper functions
// to reduce code duplication in the test suite. It relies on the generic
// `CpuTestHarness` for the underlying emulator setup.

use crate::trace_generator::NEEDED_REGISTERS;
use jolt_inlines_common::constants::{blake2, INLINE_OPCODE};
use tracer::emulator::mmu::DRAM_BASE;
use tracer::instruction::format::format_inline::FormatInline;
use tracer::instruction::{inline::INLINE, RISCVInstruction, RISCVTrace};
use tracer::utils::test_harness::CpuTestHarness;

// BLAKE2 constants
pub const HASH_STATE_SIZE: usize = 8; // 8 u64 words
pub const MESSAGE_BLOCK_SIZE: usize = 16; // 16 u64 words

/// BLAKE2-specific CPU test harness.
/// Wrapper around `CpuTestHarness` that offers convenient BLAKE2 helpers.
pub struct Blake2CpuHarness {
    pub harness: CpuTestHarness,
    pub vr: [u8; NEEDED_REGISTERS as usize],
}

impl Blake2CpuHarness {
    /// Memory layout constants
    const STATE_ADDR: u64 = DRAM_BASE;
    const MESSAGE_ADDR: u64 = DRAM_BASE + (HASH_STATE_SIZE * 8) as u64; // Separate address for message block
    const COUNTER_ADDR: u64 = Self::MESSAGE_ADDR + (MESSAGE_BLOCK_SIZE * 8) as u64; // After message block
    const FLAG_ADDR: u64 = Self::COUNTER_ADDR + 8; // After counter

    /// Register assignments for the instruction
    pub const RS1: u8 = 10; // Points to state
    pub const RS2: u8 = 11; // Points to message block + counter + final flag

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
    pub fn load_blake2_data(
        &mut self,
        state: &[u64; HASH_STATE_SIZE],
        message: &[u64; MESSAGE_BLOCK_SIZE],
        counter: u64,
        is_final: bool,
    ) {
        // Set up memory pointers in registers
        self.harness.cpu.x[Self::RS1 as usize] = Self::STATE_ADDR as i64;
        self.harness.cpu.x[Self::RS2 as usize] = Self::MESSAGE_ADDR as i64;

        // Load state into memory
        self.harness.set_memory(Self::STATE_ADDR, state);

        // Load message block into memory
        self.harness.set_memory(Self::MESSAGE_ADDR, message);

        // Load counter
        self.harness.set_memory(Self::COUNTER_ADDR, &[counter]);

        // Load final flag
        let flag_value = if is_final { 1u64 } else { 0u64 };
        self.harness.set_memory(Self::FLAG_ADDR, &[flag_value]);
    }

    /// Read the hash state from DRAM
    pub fn read_state(&mut self) -> [u64; HASH_STATE_SIZE] {
        let mut state = [0u64; HASH_STATE_SIZE];
        self.harness.read_memory(Self::STATE_ADDR, &mut state);
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
            funct3: blake2::FUNCT3,
            funct7: blake2::FUNCT7,
            inline_sequence_remaining: None,
            is_compressed: false,
        }
    }
}

impl Default for Blake2CpuHarness {
    fn default() -> Self {
        Self::new()
    }
}

pub mod blake2_verify {
    use super::*;

    pub fn print_state_hex(label: &str, state: &[u64; HASH_STATE_SIZE]) {
        println!("{label}: ");
        for (i, &word) in state.iter().enumerate() {
            println!("  [{i}]: {word:#018x}");
        }
    }

    pub fn assert_states_equal(
        expected: &[u64; HASH_STATE_SIZE],
        actual: &[u64; HASH_STATE_SIZE],
        test_name: &str,
    ) {
        if expected != actual {
            println!("\n❌ {test_name} FAILED");
            println!("\nOutputs:");
            print_state_hex("  Expected", expected);
            print_state_hex("  Actual  ", actual);
            panic!("{test_name} failed: results do not match");
        }
    }

    /// Assert that direct `exec` and virtual-sequence `trace` paths match
    pub fn assert_exec_trace_equiv(
        initial_state: &[u64; HASH_STATE_SIZE],
        message: &[u64; MESSAGE_BLOCK_SIZE],
        counter: u64,
        is_final: bool,
        expected: &[u64; HASH_STATE_SIZE],
    ) {
        let mut harness_exec = Blake2CpuHarness::new();
        let mut harness_trace = Blake2CpuHarness::new();

        harness_exec.load_blake2_data(initial_state, message, counter, is_final);
        harness_trace.load_blake2_data(initial_state, message, counter, is_final);

        let instruction = Blake2CpuHarness::instruction();

        instruction.execute(&mut harness_exec.harness.cpu, &mut ());
        instruction.trace(&mut harness_trace.harness.cpu, None);

        let exec_result = harness_exec.read_state();
        let trace_result = harness_trace.read_state();

        assert_states_equal(expected, &exec_result, "Exec result vs Expected");
        assert_states_equal(expected, &trace_result, "Trace result vs Expected");
        assert_states_equal(&exec_result, &trace_result, "Exec vs Trace equivalence");
    }
}
