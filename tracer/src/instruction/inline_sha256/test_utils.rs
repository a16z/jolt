// Test utilities for SHA256 instruction tests.
//
// This module contains SHA256-specific setup code, utilities, and helper
// functions to reduce code duplication in the test suite. It relies on the
// generic `CpuTestHarness` for the underlying emulator setup.
use super::*;
use crate::emulator::mmu::DRAM_BASE;
use crate::emulator::test_harness::CpuTestHarness;
use crate::instruction::format::format_r::FormatR;
use crate::instruction::inline_sha256::sha256::SHA256;
use crate::instruction::inline_sha256::sha256init::SHA256INIT;
use crate::instruction::{RISCVInstruction, RISCVTrace};

// Re-export canonical test vectors from test_constants module
pub use crate::instruction::inline_sha256::test_constants::TestVectors;

/// Canonical type alias for a 16-word SHA-256 input block.
pub type Sha256Block = [u32; 16];
/// Canonical type alias for an 8-word SHA-256 state/IV.
pub type Sha256State = [u32; 8];

/// SHA-256 specific CPU test harness.
/// Wrapper around `CpuTestHarness` that offers convenient SHA-256 helpers.
pub struct Sha256CpuHarness {
    pub harness: CpuTestHarness,
    // This is needed to pull out the virtual registers in case we need them for tests.
    #[allow(dead_code)]
    pub vr: [u8; NEEDED_REGISTERS as usize],
}

impl Sha256CpuHarness {
    /// Memory layout for tests.
    const BLOCK_ADDR: u64 = DRAM_BASE;
    const STATE_ADDR: u64 = DRAM_BASE + 64; // Place state right after the 64-byte block
    pub const RS1: u8 = 10;
    pub const RS2: u8 = 11;

    /// Create a new harness.
    pub fn new() -> Self {
        let vr = std::array::from_fn(|i| common::constants::virtual_register_index(i as u8));
        Self {
            // RV32.
            harness: CpuTestHarness::new_32(),
            // TODO: Fix RV64 failure (below)
            // harness: CpuTestHarness::new(),
            vr,
        }
    }

    /// Load an input block into DRAM and set `x10 = BLOCK_ADDR`.
    pub fn load_block(&mut self, block: &Sha256Block) {
        self.harness.cpu.x[Self::RS1 as usize] = Self::BLOCK_ADDR as i64;
        self.harness.set_memory32(Self::BLOCK_ADDR, block);
    }

    /// Load a state/IV into DRAM and set `x11 = STATE_ADDR`.
    pub fn load_state(&mut self, state: &Sha256State) {
        self.harness.cpu.x[Self::RS2 as usize] = Self::STATE_ADDR as i64;
        self.harness.set_memory32(Self::STATE_ADDR, state);
    }

    /// Set up output address for SHA256INIT (doesn't load initial state, just sets RS2 address).
    pub fn setup_output_only(&mut self) {
        self.harness.cpu.x[Self::RS2 as usize] = Self::STATE_ADDR as i64;
    }

    /// Read the SHA-256 state from DRAM.
    pub fn read_state(&mut self) -> Sha256State {
        let mut out = [0u32; 8];
        self.harness.read_memory32(Self::STATE_ADDR, &mut out);
        out
    }

    /// Construct a canonical SHA256 instruction.
    pub fn instruction_sha256() -> SHA256 {
        SHA256 {
            address: 0,
            operands: FormatR {
                rs1: Self::RS1,
                rs2: Self::RS2,
                rd: 0,
            },
            virtual_sequence_remaining: None,
            is_compressed: false,
        }
    }

    /// Construct a canonical SHA256INIT instruction.
    pub fn instruction_sha256init() -> SHA256INIT {
        SHA256INIT {
            address: 0,
            operands: FormatR {
                rs1: Self::RS1,
                rs2: Self::RS2,
                rd: 0,
            },
            virtual_sequence_remaining: None,
            is_compressed: false,
        }
    }
}

/// SHA-256-specific helpers for assertions.
pub mod sverify {
    use super::*;

    /// Assert two SHA-256 states are identical.
    pub fn assert_states_equal(expected: &Sha256State, actual: &Sha256State, test_name: &str) {
        if expected != actual {
            println!("\n❌ {test_name} FAILED");
            println!("Expected state: {expected:08x?}");
            println!("Actual state:   {actual:08x?}");
            panic!("{test_name} failed: states do not match");
        }
    }

    /// Assert that direct `exec` and virtual-sequence `trace` paths match for `SHA256INIT`.
    pub fn assert_exec_trace_equiv_initial(block: &Sha256Block, desc: &str) {
        let mut harness_exec = Sha256CpuHarness::new();
        let mut harness_trace = Sha256CpuHarness::new();

        // Set up both CPUs identically
        harness_exec.load_block(block);
        harness_exec.setup_output_only();
        harness_trace.load_block(block);
        harness_trace.setup_output_only();

        let instruction = Sha256CpuHarness::instruction_sha256init();

        // Execute both paths
        instruction.execute(&mut harness_exec.harness.cpu, &mut ());
        instruction.trace(&mut harness_trace.harness.cpu, None);

        // Compare results
        let exec_result = harness_exec.read_state();
        let trace_result = harness_trace.read_state();

        assert_states_equal(
            &exec_result,
            &trace_result,
            &format!("Exec vs Trace equivalence (initial): {desc}"),
        );
    }

    /// Assert that direct `exec` and virtual-sequence `trace` paths match for `SHA256`.
    pub fn assert_exec_trace_equiv_custom(block: &Sha256Block, state: &Sha256State, desc: &str) {
        let mut harness_exec = Sha256CpuHarness::new();
        let mut harness_trace = Sha256CpuHarness::new();

        // Set up both CPUs identically
        harness_exec.load_block(block);
        harness_exec.load_state(state);
        harness_trace.load_block(block);
        harness_trace.load_state(state);

        let instruction = Sha256CpuHarness::instruction_sha256();

        // Execute both paths
        instruction.execute(&mut harness_exec.harness.cpu, &mut ());
        instruction.trace(&mut harness_trace.harness.cpu, None);

        // Compare results
        let exec_result = harness_exec.read_state();
        let trace_result = harness_trace.read_state();

        assert_states_equal(
            &exec_result,
            &trace_result,
            &format!("Exec vs Trace equivalence (custom): {desc}"),
        );
    }
}
