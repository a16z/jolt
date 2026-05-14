//! Smoke test for `jolt_host::prove_program` — drives a guest ELF through
//! the modular prove pipeline + verify round-trip.
//!
//! Uses `muldiv` from `examples/muldiv` because it's the smallest, lowest-
//! dep guest that fits the fixture shape (max_trace_length = 65536,
//! i.e. log_t = 16). No rayon, no String allocator, just `a * b / c`.
//!
//! Requires the `jolt` CLI to be installed (`cargo install --path .`).

#![expect(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr
)]

use jolt_host::{prove_program, verify_proof, ProveProgramError};
use jolt_trace::Program;

#[test]
#[ignore = "exercises full modular prove pipeline; requires jolt CLI"]
fn muldiv_modular_prove_smoke() {
    let mut program = Program::new("muldiv-guest");
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("postcard encode muldiv inputs");

    let result = prove_program(&mut program, &inputs, &[], &[]);

    match result {
        Ok(output) => {
            eprintln!(
                "[prove_program_smoke] proof generated, io output bytes = {}, \
                 evaluation present = {}",
                output.io_device.outputs.len(),
                output.proof.evaluation.is_some()
            );
            verify_proof(&output, &mut program).expect("modular verifier accepts muldiv proof");
        }
        Err(ProveProgramError::UnsupportedShape {
            log_t,
            log_k_bytecode,
            log_k_ram,
        }) => {
            panic!(
                "muldiv shape mismatch: actual=({log_t}, {log_k_bytecode}, {log_k_ram}), \
                 fixture=(16, 10, 16). Goldens need regen at the new shape."
            );
        }
        Err(other) => {
            panic!("prove_program failed unexpectedly: {other:?}");
        }
    }
}
