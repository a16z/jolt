//! Generic test harness for the RISC-V CPU emulator.
//!
//! Provides a controlled environment for testing instructions by setting up a
//! CPU instance with memory and offering helper functions for execution and state assertion.

use crate::emulator::{cpu::Cpu, default_terminal::DefaultTerminal};
use crate::instruction::{RAMRead, RAMWrite, RISCVInstruction, RV32IMInstruction};

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
    /// Creates a new test harness with an initialized CPU and memory.
    pub fn new() -> Self {
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        Self { cpu }
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

/// Generic instruction executor that handles instruction types used in custom instructions.
pub fn execute_instruction(cpu: &mut Cpu, instr: &RV32IMInstruction) {
    match instr {
        RV32IMInstruction::LD(ld) => {
            let mut ram_access = RAMRead::default();
            ld.execute(cpu, &mut ram_access);
        }
        RV32IMInstruction::SD(sd) => {
            let mut ram_access = RAMWrite::default();
            sd.execute(cpu, &mut ram_access);
        }
        RV32IMInstruction::XOR(xor) => {
            let mut ram_access = ();
            xor.execute(cpu, &mut ram_access);
        }
        RV32IMInstruction::XORI(xori) => {
            let mut ram_access = ();
            xori.execute(cpu, &mut ram_access);
        }
        RV32IMInstruction::AND(and) => {
            let mut ram_access = ();
            and.execute(cpu, &mut ram_access);
        }
        RV32IMInstruction::ANDI(andi) => {
            let mut ram_access = ();
            andi.execute(cpu, &mut ram_access);
        }
        RV32IMInstruction::OR(or) => {
            let mut ram_access = ();
            or.execute(cpu, &mut ram_access);
        }
        RV32IMInstruction::SLLI(slli) => {
            let mut ram_access = ();
            slli.execute(cpu, &mut ram_access);
        }
        RV32IMInstruction::SRLI(srli) => {
            let mut ram_access = ();
            srli.execute(cpu, &mut ram_access);
        }
        RV32IMInstruction::VirtualROTRI(rotri) => {
            let mut ram_access = ();
            rotri.execute(cpu, &mut ram_access);
        }
        RV32IMInstruction::VirtualMULI(muli) => {
            let mut ram_access = ();
            muli.execute(cpu, &mut ram_access);
        }
        RV32IMInstruction::ADDI(addi) => {
            let mut ram_access = ();
            addi.execute(cpu, &mut ram_access);
        }
        RV32IMInstruction::LUI(lui) => {
            let mut ram_access = ();
            lui.execute(cpu, &mut ram_access);
        }
        _ => panic!("Unexpected instruction type: {:?}", instr),
    }
}
