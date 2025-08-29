// Test utilities for BigInt multiplication instruction tests
//
// This module contains BigInt-specific setup code, utilities, and helper functions
// to reduce code duplication in the test suite. It relies on the generic
// `CpuTestHarness` for the underlying emulator setup.

use super::{INPUT_LIMBS, OUTPUT_LIMBS};
use crate::multiplication::trace_generator::NEEDED_REGISTERS;

use tracer::emulator::mmu::DRAM_BASE;
use tracer::instruction::format::format_r::FormatR;
use tracer::instruction::{inline::INLINE, RISCVInstruction, RISCVTrace};
use tracer::utils::test_harness::CpuTestHarness;

pub type BigIntInput = ([u64; INPUT_LIMBS], [u64; INPUT_LIMBS]);
pub type BigIntOutput = [u64; OUTPUT_LIMBS];

/// BigInt-specific CPU test harness.
/// Wrapper around `CpuTestHarness` that offers convenient BigInt helpers.
pub struct BigIntCpuHarness {
    pub harness: CpuTestHarness,
    pub vr: [u8; NEEDED_REGISTERS],
}

impl BigIntCpuHarness {
    /// Memory layout constants
    const BASE_ADDR: u64 = DRAM_BASE;
    const LHS_OFFSET: u64 = 0;
    const RHS_OFFSET: u64 = (INPUT_LIMBS * 8) as u64; // 4 * 8 bytes
    const RESULT_OFFSET: u64 = (INPUT_LIMBS * 2 * 8) as u64; // 8 * 8 bytes

    /// Register assignments for the instruction
    pub const RS1: u8 = 10; // Address of first operand
    pub const RS2: u8 = 11; // Address of second operand
    pub const RD: u8 = 12; // Address where result will be stored

    /// Create a new harness with initialized memory.
    pub fn new() -> Self {
        // Allocate virtual registers
        let guards: Vec<_> = (0..NEEDED_REGISTERS)
            .map(|_| tracer::utils::virtual_registers::allocate_virtual_register())
            .collect();
        let vr: [u8; 20] = core::array::from_fn(|i| *guards[i]);

        Self {
            harness: CpuTestHarness::new(),
            vr,
        }
    }

    /// Load operands into DRAM and set up registers
    pub fn load_operands(&mut self, lhs: &[u64; INPUT_LIMBS], rhs: &[u64; INPUT_LIMBS]) {
        // Set up memory pointers in registers
        self.harness.cpu.x[Self::RS1 as usize] = (Self::BASE_ADDR + Self::LHS_OFFSET) as i64;
        self.harness.cpu.x[Self::RS2 as usize] = (Self::BASE_ADDR + Self::RHS_OFFSET) as i64;
        self.harness.cpu.x[Self::RD as usize] = (Self::BASE_ADDR + Self::RESULT_OFFSET) as i64;

        // Load operands into memory
        self.harness
            .set_memory(Self::BASE_ADDR + Self::LHS_OFFSET, lhs);
        self.harness
            .set_memory(Self::BASE_ADDR + Self::RHS_OFFSET, rhs);
    }

    /// Read the multiplication result from DRAM
    pub fn read_result(&mut self) -> BigIntOutput {
        let mut out = [0u64; OUTPUT_LIMBS];
        self.harness
            .read_memory(Self::BASE_ADDR + Self::RESULT_OFFSET, &mut out);
        out
    }

    /// Construct a canonical BIGINT256_MUL instruction
    pub fn instruction() -> INLINE {
        INLINE {
            address: 0,
            operands: FormatR {
                rs1: Self::RS1,
                rs2: Self::RS2,
                rd: Self::RD,
            },
            // BIGINT256_MUL has opcode 0x0B, funct3 0x00, funct7 0x02
            opcode: 0x0B,
            funct3: 0x00,
            funct7: 0x02,
            inline_sequence_remaining: None,
            is_compressed: false,
        }
    }
}

impl Default for BigIntCpuHarness {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper module for BigInt-specific assertions
pub mod bigint_verify {
    use super::*;

    /// Print a BigInt value in hexadecimal format
    pub fn print_limbs_hex(label: &str, limbs: &[u64]) {
        print!("{label}: 0x");
        // Print in big-endian order (most significant limb first)
        for i in (0..limbs.len()).rev() {
            print!("{:016x}", limbs[i]);
            if i > 0 {
                print!("_");
            }
        }
        println!();
    }

    /// Assert two BigInt values are identical
    pub fn assert_bigints_equal(
        expected: &BigIntOutput,
        actual: &BigIntOutput,
        lhs: &[u64; INPUT_LIMBS],
        rhs: &[u64; INPUT_LIMBS],
        test_name: &str,
    ) {
        if expected != actual {
            println!("\n❌ {test_name} FAILED");
            println!("\nInputs:");
            print_limbs_hex("  LHS    ", lhs);
            print_limbs_hex("  RHS    ", rhs);
            println!("\nOutputs:");
            print_limbs_hex("  Expected", expected);
            print_limbs_hex("  Actual  ", actual);

            // Show limb-by-limb comparison
            println!("\nDifferences:");
            for i in 0..OUTPUT_LIMBS {
                if expected[i] != actual[i] {
                    println!(
                        "  Limb {}: expected 0x{:016x}, got 0x{:016x}",
                        i, expected[i], actual[i]
                    );
                }
            }
            panic!("{test_name} failed: results do not match");
        }
    }

    /// Assert that direct `exec` and virtual-sequence `trace` paths match
    pub fn assert_exec_trace_equiv(
        lhs: &[u64; INPUT_LIMBS],
        rhs: &[u64; INPUT_LIMBS],
        expected: &[u64; OUTPUT_LIMBS],
    ) {
        let mut harness_exec = BigIntCpuHarness::new();
        let mut harness_trace = BigIntCpuHarness::new();

        // Set up both CPUs identically
        harness_exec.load_operands(lhs, rhs);
        harness_trace.load_operands(lhs, rhs);

        let instruction = BigIntCpuHarness::instruction();

        // Execute both paths
        instruction.execute(&mut harness_exec.harness.cpu, &mut ());
        instruction.trace(&mut harness_trace.harness.cpu, None);

        // Compare results
        let exec_result = harness_exec.read_result();
        let trace_result = harness_trace.read_result();

        assert_bigints_equal(&exec_result, expected, lhs, rhs, "Exec result vs Expected");
        assert_bigints_equal(
            &trace_result,
            expected,
            lhs,
            rhs,
            "Trace result vs Expected",
        );
        assert_bigints_equal(
            &exec_result,
            &trace_result,
            lhs,
            rhs,
            "Exec vs Trace equivalence",
        );
    }
}
