use crate::sequence_builder::Keccak256Permutation;
use crate::test_constants::{self, TestVectors};
use crate::Keccak256State;
use jolt_inlines_sdk::spec::InlineSpec;

pub struct KeccakTestCase {
    pub input: Keccak256State,
    pub expected: Keccak256State,
    pub description: &'static str,
}

pub fn keccak_test_vectors() -> Vec<KeccakTestCase> {
    vec![
        KeccakTestCase {
            input: [0u64; 25],
            expected: test_constants::xkcp_vectors::AFTER_ONE_PERMUTATION,
            description: "All zeros input (XKCP test vector)",
        },
        KeccakTestCase {
            input: TestVectors::create_simple_pattern(),
            expected: Keccak256Permutation::reference(&TestVectors::create_simple_pattern()),
            description: "Simple arithmetic pattern",
        },
        KeccakTestCase {
            input: {
                let mut state = [0u64; 25];
                state[0] = 1;
                state
            },
            expected: {
                let mut state = [0u64; 25];
                state[0] = 1;
                Keccak256Permutation::reference(&state)
            },
            description: "Single bit in first lane",
        },
    ]
}

pub fn print_state_hex(state: &Keccak256State) {
    for (i, &lane) in state.iter().enumerate() {
        if i % 5 == 0 {
            println!();
        }
        print!("{lane:#018x} ");
    }
    println!();
}

pub mod kverify {
    use super::*;

    pub fn assert_states_equal(
        expected: &Keccak256State,
        actual: &Keccak256State,
        test_name: &str,
    ) {
        if expected != actual {
            println!("\n{test_name} FAILED");
            println!("Expected state:");
            print_state_hex(expected);
            println!("Actual state:");
            print_state_hex(actual);

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
