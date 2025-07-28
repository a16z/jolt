//! Generic test harness for the RISC-V CPU emulator.
//!
//! Provides a controlled environment for testing instructions by setting up a
//! CPU instance with memory and offering helper functions for execution and state assertion.

use crate::emulator::{
    cpu::{Cpu, Xlen},
    default_terminal::DefaultTerminal,
};
use crate::instruction::{RISCVInstruction, RV32IMInstruction};

/// Default memory capacity for tests.
pub const TEST_MEMORY_CAPACITY: u64 = 1024 * 1024; // 1MB

/// A test harness for the RISC-V CPU emulator.
///
/// This struct encapsulates a `Cpu` instance, providing a controlled
/// environment for setting up, executing, and asserting on instruction behavior.
pub struct CpuTestHarness {
    pub cpu: Cpu,
}

impl CpuTestHarness {
    /// Creates a harness with RV64.
    pub fn new() -> Self {
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        Self { cpu }
    }

    /// Creates a harness with RV32.
    pub fn new_32() -> Self {
        let mut h = Self::new();
        h.cpu.update_xlen(Xlen::Bit32);
        h
    }

    /// Sets a region of the harness's memory with the provided data.
    pub fn set_memory(&mut self, base_addr: u64, data: &[u64]) {
        for (i, &word) in data.iter().enumerate() {
            self.cpu
                .mmu
                .store_doubleword(base_addr + (i * 8) as u64, word)
                .expect("Failed to store word to memory");
        }
    }

    pub fn read_memory(&mut self, base_addr: u64, data: &mut [u64]) {
        for (i, lane) in data.iter_mut().enumerate() {
            *lane = self
                .cpu
                .mmu
                .load_doubleword(base_addr + (i * 8) as u64)
                .expect("Failed to load word from memory")
                .0;
        }
    }

    /// Reads CPU register values specified by indices into a provided mutable slice.
    ///
    /// # Panics
    /// Panics if the length of `register_indices` does not match the length of `output`.
    pub fn read_registers(&self, register_indices: &[usize], output: &mut [u64]) {
        assert_eq!(
            register_indices.len(),
            output.len(),
            "Register indices and output buffer must have the same length"
        );
        for (i, &reg_index) in register_indices.iter().enumerate() {
            output[i] = self.cpu.x[reg_index] as u64;
        }
    }

    pub fn execute_virtual_sequence(&mut self, sequence: &[RV32IMInstruction]) {
        for instr in sequence {
            instr.execute(&mut self.cpu);
        }
    }

    /// Encodes an R-type instruction by filling in rd, rs1, rs2 fields.
    /// `template` should be the instruction's MATCH constant containing funct7/funct3/opcode bits.
    pub fn encode_r(template: u32, rd: usize, rs1: usize, rs2: usize) -> u32 {
        template | ((rs2 as u32) << 20) | ((rs1 as u32) << 15) | ((rd as u32) << 7)
    }

    fn exec_word(mut self, rs1_val: u64, rs2_val: u64, instr_word: u32) -> u64 {
        self.cpu.x[1] = rs1_val as i64;
        self.cpu.x[2] = rs2_val as i64;
        RV32IMInstruction::decode(instr_word, 0)
            .expect("decode")
            .execute(&mut self.cpu);
        self.cpu.x[3] as u64
    }

    /// Execute an R-type instruction template using standard registers x1, x2 -> x3.
    pub fn exec_r<I: RISCVInstruction>(self, rs1_val: u64, rs2_val: u64) -> u64 {
        let word = Self::encode_r(I::MATCH, 3, 1, 2);
        self.exec_word(rs1_val, rs2_val, word)
    }

    /// Execute an R-type instruction template with custom rd/rs1/rs2 indices.
    pub fn exec_r_with<I: RISCVInstruction>(
        self,
        rd: usize,
        rs1_idx: usize,
        rs2_idx: usize,
        rs1_val: u64,
        rs2_val: u64,
    ) -> u64 {
        let word = Self::encode_r(I::MATCH, rd, rs1_idx, rs2_idx);
        let mut harness = self;
        harness.cpu.x[rs1_idx] = rs1_val as i64;
        harness.cpu.x[rs2_idx] = rs2_val as i64;
        RV32IMInstruction::decode(word, 0)
            .expect("decode")
            .execute(&mut harness.cpu);
        harness.cpu.x[rd] as u64
    }
}

impl Default for CpuTestHarness {
    fn default() -> Self {
        Self::new()
    }
}

/// A generic test case for instruction testing.
#[derive(Debug)]
pub struct InstructionTestCase<I, E> {
    pub input: I,
    pub expected: E,
    pub description: &'static str,
}

impl<I, E> InstructionTestCase<I, E> {
    /// Creates a new test case.
    pub fn new(input: I, expected: E, description: &'static str) -> Self {
        Self {
            input,
            expected,
            description,
        }
    }
}
