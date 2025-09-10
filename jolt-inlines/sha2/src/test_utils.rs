// Test utilities for SHA256 instruction tests.
//
// This module contains SHA256-specific setup code, utilities, and helper
// functions to reduce code duplication in the test suite. It relies on the
// generic `CpuTestHarness` for the underlying emulator setup.
use crate::trace_generator::NEEDED_REGISTERS;
use tracer::emulator::cpu::Xlen;
use tracer::emulator::mmu::DRAM_BASE;
use tracer::instruction::format::format_inline::FormatInline;
use tracer::instruction::inline::INLINE;
use tracer::utils::test_harness::CpuTestHarness;

/// Canonical type alias for a 16-word SHA-256 input block.
pub type Sha256Block = [u32; 16];
/// Canonical type alias for an 8-word SHA-256 state/IV.
pub type Sha256State = [u32; 8];

/// SHA-256 specific CPU test harness.
/// Wrapper around `CpuTestHarness` that offers convenient SHA-256 helpers.
pub struct Sha256CpuHarness {
    pub harness: CpuTestHarness,
    // This is needed to pull out the virtual registers in case we need them for tests.
    pub vr: [u8; NEEDED_REGISTERS as usize],
}

impl Sha256CpuHarness {
    /// Memory layout for tests.
    const BLOCK_ADDR: u64 = DRAM_BASE;
    const STATE_ADDR: u64 = DRAM_BASE + 64; // Place state right after the 64-byte block
    pub const RS1: u8 = 10;
    pub const RS2: u8 = 11;

    /// Create a new harness.
    pub fn new(xlen: Xlen) -> Self {
        let guards: Vec<_> = (0..32)
            .map(|_| tracer::utils::virtual_registers::allocate_virtual_register_for_inline())
            .collect();
        let vr: [u8; 32] = std::array::from_fn(|i| *guards[i]);
        Self {
            harness: if xlen == Xlen::Bit32 {
                CpuTestHarness::new_32()
            } else {
                CpuTestHarness::new()
            },
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
    pub fn instruction_sha256() -> INLINE {
        INLINE {
            address: 0,
            operands: FormatInline {
                rs1: Self::RS1,
                rs2: Self::RS2,
                rs3: 0,
            },
            // SHA256 has opcode 0x0B, funct3 0x00, funct7 0x00
            opcode: 0x0B,
            funct3: 0x00,
            funct7: 0x00,
            inline_sequence_remaining: None,
            is_compressed: false,
        }
    }

    /// Construct a canonical SHA256INIT instruction.
    pub fn instruction_sha256init() -> INLINE {
        INLINE {
            address: 0,
            operands: FormatInline {
                rs1: Self::RS1,
                rs2: Self::RS2,
                rs3: 0,
            },
            // SHA256INIT has opcode 0x0B, funct3 0x01, funct7 0x00
            opcode: 0x0B,
            funct3: 0x01,
            funct7: 0x00,
            inline_sequence_remaining: None,
            is_compressed: false,
        }
    }

    /// Execute an inline sequence of instructions.
    pub fn execute_inline_sequence(&mut self, sequence: &[tracer::instruction::RV32IMInstruction]) {
        self.harness.execute_inline_sequence(sequence);
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
}
