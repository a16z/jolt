// Test utilities for Keccak256 instruction tests
//
// This module contains Keccak-specific setup code, utilities, and helper functions
// to reduce code duplication in the test suite. It relies on the generic
// `CpuTestHarness` for the underlying emulator setup.
use super::*;
use crate::emulator::mmu::DRAM_BASE;
use crate::emulator::test_harness::{CpuTestHarness, InstructionTestCase};
use crate::instruction::format::format_r::FormatR;
use crate::instruction::inline_keccak256::keccak256::KECCAK256;
use crate::instruction::inline_keccak256::test_constants::{self, TestVectors};
use crate::instruction::inline_keccak256::NEEDED_REGISTERS;
use crate::instruction::inline_keccak256::{
    execute_chi, execute_iota, execute_keccak_f, execute_rho_and_pi, execute_theta, ROUND_CONSTANTS,
};
use crate::instruction::{RISCVInstruction, RISCVTrace};

/// Canonical type alias for a 25-lane Keccak state.
pub type Keccak256State = [u64; 25];

pub type KeccakTestCase = InstructionTestCase<Keccak256State, Keccak256State>;

/// Keccak-specific CPU test harness.
/// Wrapper around `CpuTestHarness` that offers convenient Keccak helpers.
pub struct KeccakCpuHarness {
    pub harness: CpuTestHarness,
    pub vr: [usize; NEEDED_REGISTERS],
}

impl KeccakCpuHarness {
    /// Virtual register layout used by the Keccak virtual sequence.
    const BASE_ADDR: u64 = DRAM_BASE;
    pub const RS1: usize = 10;

    /// Create a new harness with initialized memory.
    pub fn new() -> Self {
        let vr =
            core::array::from_fn(|i| common::constants::virtual_register_index(i as u64) as usize);

        Self {
            harness: CpuTestHarness::new(),
            vr,
        }
    }

    /// Load a Keccak state into DRAM and set `x10 = BASE_ADDR`.
    pub fn load_state(&mut self, state: &Keccak256State) {
        self.harness.cpu.x[Self::RS1] = Self::BASE_ADDR as i64;
        self.harness.set_memory(Self::BASE_ADDR, state);
    }

    /// Read the Keccak state from DRAM.
    pub fn read_state(&mut self) -> Keccak256State {
        let mut out = [0u64; 25];
        self.harness.read_memory(Self::BASE_ADDR, &mut out);
        out
    }

    /// Read the Keccak virtual registers used to store the state (first 25).
    pub fn read_vr(&self) -> Keccak256State {
        let mut out = [0u64; 25];
        self.harness.read_registers(&self.vr[..25], &mut out);
        out
    }

    pub fn read_vr_at_offset(&self, offset: usize) -> Keccak256State {
        let mut out = [0u64; 25];
        self.harness
            .read_registers(&self.vr[offset..offset + 25], &mut out);
        out
    }

    /// Construct a canonical KECCAK256 instruction.
    pub fn instruction() -> KECCAK256 {
        KECCAK256 {
            address: 0,
            operands: FormatR {
                rs1: Self::RS1,
                rs2: 0,
                rd: 0,
            },
            virtual_sequence_remaining: None,
            is_compressed: false,
        }
    }

    pub fn execute_keccak_instruction(&mut self) {
        let instruction = Self::instruction();
        instruction.execute(&mut self.harness.cpu, &mut ());
    }

    pub fn trace_keccak_instruction(&mut self) -> Vec<crate::instruction::RV32IMCycle> {
        let instruction = Self::instruction();
        let mut trace = Vec::new();
        instruction.trace(&mut self.harness.cpu, Some(&mut trace));
        trace
    }

    pub fn execute_virtual_sequence(&mut self, sequence: &[RV32IMInstruction]) {
        self.harness.execute_virtual_sequence(sequence);
    }
}

impl Default for KeccakCpuHarness {
    fn default() -> Self {
        Self::new()
    }
}

/// Create test cases for direct execution testing.
pub fn keccak_test_vectors() -> Vec<KeccakTestCase> {
    vec![
        // Test case 1: All zeros input (standard test vector)
        KeccakTestCase::new(
            [0u64; 25],
            test_constants::xkcp_vectors::AFTER_ONE_PERMUTATION,
            "All zeros input (XKCP test vector)",
        ),
        // Test case 2: Simple pattern
        KeccakTestCase::new(
            TestVectors::create_simple_pattern(),
            {
                let mut state = TestVectors::create_simple_pattern();
                execute_keccak_f(&mut state);
                state
            },
            "Simple arithmetic pattern",
        ),
        // Test case 3: Single bit set
        KeccakTestCase::new(
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
/// Print a Keccak state in hex format for debugging.
pub fn print_state_hex(state: &Keccak256State) {
    for (i, &lane) in state.iter().enumerate() {
        if i % 5 == 0 {
            println!();
        }
        print!("{lane:#018x} ");
    }
    println!();
}

/// Execute reference implementation up to a specific step in a specific round.
pub fn execute_reference_up_to_step(
    initial_state: &Keccak256State,
    target_round: usize,
    target_step: &str,
) -> Keccak256State {
    let mut state = *initial_state;

    for (round, constant) in ROUND_CONSTANTS.iter().enumerate().take(target_round + 1) {
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

        execute_iota(&mut state, *constant);
        if round == target_round && target_step == "iota" {
            break;
        }
    }

    state
}

/// Keccak-specific helpers for assertions.
pub mod kverify {
    use super::*;

    /// Assert two Keccak states are identical.
    pub fn assert_states_equal(
        expected: &Keccak256State,
        actual: &Keccak256State,
        test_name: &str,
    ) {
        if expected != actual {
            println!("\n❌ {test_name} FAILED");
            println!("Expected state:");
            print_state_hex(expected);
            println!("Actual state:");
            print_state_hex(actual);

            // Show first few mismatches
            let mut mismatch_count = 0;
            for i in 0..25 {
                if expected[i] != actual[i] {
                    println!(
                        "  Lane {i}: expected 0x{:016x}, got 0x{:016x}",
                        expected[i], actual[i]
                    );
                    mismatch_count += 1;
                    if mismatch_count >= 5 {
                        println!("  ... (showing first 5 mismatches)");
                        break;
                    }
                }
            }
            panic!("{test_name} failed: states do not match");
        }
    }

    /// Assert that direct `exec` and virtual-sequence `trace` paths match.
    pub fn assert_exec_trace_equiv(initial_state: &Keccak256State, desc: &str) {
        let mut harness_exec = KeccakCpuHarness::new();
        let mut harness_trace = KeccakCpuHarness::new();

        // Set up both CPUs identically
        harness_exec.load_state(initial_state);
        harness_trace.load_state(initial_state);

        let instruction = KeccakCpuHarness::instruction();

        // Execute both paths
        instruction.execute(&mut harness_exec.harness.cpu, &mut ());
        instruction.trace(&mut harness_trace.harness.cpu, None);

        // Compare results
        let exec_result = harness_exec.read_state();
        let trace_result = harness_trace.read_state();

        self::assert_states_equal(
            &exec_result,
            &trace_result,
            &format!("Exec vs Trace equivalence: {desc}"),
        );
    }
}
