use crate::exec::execute_keccak_f;
use crate::test_constants::{self, TestVectors};
use crate::Keccak256State;
use tracer::emulator::cpu::Xlen;
use tracer::instruction::inline::INLINE;
use tracer::utils::inline_test_harness::{InlineMemoryLayout, InlineTestHarness};
use tracer::utils::test_harness::InstructionTestCase;

pub type KeccakTestCase = InstructionTestCase<Keccak256State, Keccak256State>;

pub fn create_keccak_harness(xlen: Xlen) -> InlineTestHarness {
    // Keccak256: rs1=state/output, rs2=input
    let layout = InlineMemoryLayout::single_input(136, 200); // 136-byte block, 200-byte state
    InlineTestHarness::new(layout, xlen)
}

pub fn instruction() -> INLINE {
    InlineTestHarness::create_default_instruction(
        crate::INLINE_OPCODE,
        crate::KECCAK256_FUNCT3,
        crate::KECCAK256_FUNCT7,
    )
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
}
