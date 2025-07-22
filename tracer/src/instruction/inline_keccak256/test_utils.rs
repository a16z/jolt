//! Test utilities for Keccak256 instruction tests
//!
//! This module contains common setup code, utilities, and helper functions
//! to reduce code duplication in the test suite.
#[cfg(test)]
use super::*;
use crate::emulator::{cpu::Cpu, default_terminal::DefaultTerminal, mmu::DRAM_BASE};
use crate::instruction::format::format_r::FormatR;
use crate::instruction::inline_keccak256::keccak256::KECCAK256;
use crate::instruction::inline_keccak256::test_constants::TestVectors;
use crate::instruction::inline_keccak256::{
    execute_chi, execute_iota, execute_keccak_f, execute_rho_and_pi, execute_theta,
    NEEDED_REGISTERS, ROUND_CONSTANTS,
};
use crate::instruction::{RAMRead, RAMWrite, RISCVInstruction, RISCVTrace, RV32IMInstruction};
use common::constants::virtual_register_index;

/// Test memory capacity used across all tests
pub const TEST_MEMORY_CAPACITY: u64 = 1024 * 1024; // 1MB

/// Test setup for Keccak256 instruction testing
pub struct KeccakTestSetup {
    pub cpu: Cpu,
    pub base_addr: u64,
    pub vr: [usize; NEEDED_REGISTERS],
}

impl KeccakTestSetup {
    /// Create a new test setup with initialized CPU and virtual registers
    pub fn new() -> Self {
        let mut cpu = Cpu::new(Box::new(DefaultTerminal::new()));
        cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
        let base_addr = DRAM_BASE;

        // Initialize virtual register mapping
        let mut vr = [0; NEEDED_REGISTERS];
        for i in 0..NEEDED_REGISTERS {
            vr[i] = virtual_register_index(i as u64) as usize;
        }

        Self { cpu, base_addr, vr }
    }

    /// Load a Keccak state into memory and set up rs1 register
    pub fn load_state_to_memory(&mut self, state: &[u64; 25]) {
        self.cpu.x[10] = self.base_addr as i64;
        for (i, &lane) in state.iter().enumerate() {
            self.cpu
                .mmu
                .store_doubleword(self.base_addr + (i * 8) as u64, lane)
                .expect("Failed to store lane to memory");
        }
    }

    /// Read Keccak state from memory
    pub fn read_state_from_memory(&mut self) -> [u64; 25] {
        let mut state = [0u64; 25];
        for (i, lane) in state.iter_mut().enumerate() {
            *lane = self
                .cpu
                .mmu
                .load_doubleword(self.base_addr + (i * 8) as u64)
                .expect("Failed to load lane from memory")
                .0;
        }
        state
    }

    /// Read Keccak state from virtual registers
    pub fn read_state_from_registers(&self) -> [u64; 25] {
        let mut state = [0u64; 25];
        for i in 0..25 {
            state[i] = self.cpu.x[self.vr[i]] as u64;
        }
        state
    }

    /// Create a standard KECCAK256 instruction for testing
    pub fn create_instruction(&self) -> KECCAK256 {
        KECCAK256 {
            address: 0,
            operands: FormatR {
                rs1: 10,
                rs2: 0,
                rd: 0,
            },
            virtual_sequence_remaining: None,
        }
    }

    /// Execute a virtual instruction sequence step by step
    pub fn execute_virtual_sequence(&mut self, sequence: &[RV32IMInstruction]) {
        for instr in sequence {
            execute_instruction(&mut self.cpu, instr);
        }
    }
}

/// Generic instruction executor that handles all instruction types used in Keccak256
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

/// Print a Keccak state in hex format for debugging
pub fn print_state_hex(state: &[u64; 25]) {
    for (i, &lane) in state.iter().enumerate() {
        if i % 5 == 0 {
            println!();
        }
        print!("{:#018x} ", lane);
    }
    println!();
}

/// Compare two states and print differences
pub fn compare_states(expected: &[u64; 25], actual: &[u64; 25], description: &str) -> bool {
    let mut all_match = true;
    for i in 0..25 {
        if expected[i] != actual[i] {
            println!(
                "{} mismatch at lane {}: expected {:#018x}, got {:#018x}",
                description, i, expected[i], actual[i]
            );
            all_match = false;
        }
    }
    all_match
}

/// Execute reference implementation up to a specific step in a specific round
pub fn execute_reference_up_to_step(
    initial_state: &[u64; 25],
    target_round: usize,
    target_step: &str,
) -> [u64; 25] {
    let mut state = *initial_state;

    for round in 0..=target_round {
        execute_theta(&mut state);
        if round == target_round && target_step == "theta" {
            break;
        }

        execute_rho_and_pi(&mut state);
        if round == target_round && target_step == "rho_and_pi" {
            break;
        }

        execute_chi(&mut state);
        if round == target_round && target_step == "chi" {
            break;
        }

        execute_iota(&mut state, ROUND_CONSTANTS[round]);
        if round == target_round && target_step == "iota" {
            break;
        }
    }

    state
}

/// Test case structure for parameterized tests
#[derive(Debug)]
pub struct TestCase {
    pub input: [u64; 25],
    pub expected: [u64; 25],
    pub description: &'static str,
}

impl TestCase {
    /// Create a new test case
    pub fn new(input: [u64; 25], expected: [u64; 25], description: &'static str) -> Self {
        Self {
            input,
            expected,
            description,
        }
    }

    /// Create test cases for direct execution testing
    /// TODO: Refactor out as this is Keccak256 specific. Might want to move to test_constants.rs
    pub fn create_direct_execution_test_cases() -> Vec<TestCase> {
        vec![
            // Test case 1: All zeros input (standard test vector)
            TestCase::new(
                [0u64; 25],
                test_constants::xkcp_vectors::AFTER_ONE_PERMUTATION,
                "All zeros input (XKCP test vector)",
            ),
            // Test case 2: Simple pattern
            TestCase::new(
                TestVectors::create_simple_pattern(),
                {
                    let mut state = TestVectors::create_simple_pattern();
                    execute_keccak_f(&mut state);
                    state
                },
                "Simple arithmetic pattern",
            ),
            // Test case 3: Single bit set
            TestCase::new(
                {
                    let mut state = [0u64; 25];
                    state[0] = 1;
                    state
                },
                {
                    let mut state = [0u64; 25];
                    state[0] = 1;
                    execute_keccak_f(&mut state);
                    state
                },
                "Single bit in first lane",
            ),
        ]
    }
}

/// Assertion helpers for cleaner test code
pub struct TestAssertions;

impl TestAssertions {
    /// Assert that two states are equal with a detailed error message
    pub fn assert_states_equal(expected: &[u64; 25], actual: &[u64; 25], test_name: &str) {
        if expected != actual {
            println!("\n❌ {} FAILED", test_name);
            println!("Expected state:");
            print_state_hex(expected);
            println!("Actual state:");
            print_state_hex(actual);

            // Show first few mismatches
            let mut mismatch_count = 0;
            for i in 0..25 {
                if expected[i] != actual[i] {
                    println!(
                        "  Lane {}: expected 0x{:016x}, got 0x{:016x}",
                        i, expected[i], actual[i]
                    );
                    mismatch_count += 1;
                    if mismatch_count >= 5 {
                        println!("  ... (showing first 5 mismatches)");
                        break;
                    }
                }
            }

            panic!("{} failed: states do not match", test_name);
        }
    }

    /// Assert that exec and trace paths produce the same result
    pub fn assert_exec_trace_equivalence(initial_state: &[u64; 25], description: &str) {
        let mut setup_exec = KeccakTestSetup::new();
        let mut setup_trace = KeccakTestSetup::new();

        // Set up both CPUs identically
        setup_exec.load_state_to_memory(initial_state);
        setup_trace.load_state_to_memory(initial_state);

        let instruction = setup_exec.create_instruction();

        // Execute both paths
        instruction.execute(&mut setup_exec.cpu, &mut ());
        instruction.trace(&mut setup_trace.cpu, None);

        // Compare results
        let exec_result = setup_exec.read_state_from_memory();
        let trace_result = setup_trace.read_state_from_memory();

        Self::assert_states_equal(
            &exec_result,
            &trace_result,
            &format!("Exec vs Trace equivalence: {}", description),
        );
    }
}
