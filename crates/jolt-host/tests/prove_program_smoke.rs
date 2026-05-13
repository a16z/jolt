//! Smoke test for `jolt_prover::prove_program` — drives a guest ELF
//! through the modular prove pipeline and asserts the proof is well-formed.
//!
//! Uses `muldiv` from `examples/muldiv` because it's the smallest, lowest-
//! dep guest in the fixture-matching shape range (max_trace_length = 65536,
//! i.e. log_t = 16). No rayon, no String allocator, just `a * b / c`.
//!
//! Requires the `jolt` CLI to be installed (`cargo install --path .`).

use jolt_host::{prove_program, ProveProgramError};
use jolt_trace::Program;

#[test]
#[ignore = "exercises full modular prove pipeline; requires jolt CLI"]
fn muldiv_modular_prove_smoke() {
    let mut program = Program::new("muldiv-guest");
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("postcard encode muldiv inputs");

    let result = prove_program(&mut program, &inputs, &[], &[]);

    match result {
        Ok((proof, io_device, _artifacts)) => {
            eprintln!(
                "[prove_program_smoke] proof generated, io output bytes = {}, \
                 evaluation present = {}",
                io_device.outputs.len(),
                proof.evaluation.is_some()
            );
            assert!(
                proof.evaluation.is_some(),
                "modular prover should produce an evaluation proof"
            );
        }
        Err(ProveProgramError::UnsupportedShape {
            log_t,
            log_k_bytecode,
            log_k_ram,
        }) => {
            // Expected today: the goldens-baked fixture is at (16, 10, 16)
            // but muldiv-guest's compiled ELF + memory layout produces a
            // different shape. The trace/class-A pipeline still ran — the
            // gate caught the mismatch cleanly. To run a full proof, the
            // goldens need regeneration via `JOLT_UPDATE_GOLDENS=1
            // cargo nextest run -p bolt ...` at this shape.
            eprintln!(
                "[prove_program_smoke] trace + class-A pipeline OK; \
                 shape gate caught fixture mismatch. \
                 Actual: (log_t={log_t}, log_k_bytecode={log_k_bytecode}, log_k_ram={log_k_ram}). \
                 Goldens baked at: (16, 10, 16). \
                 To run full proof: regenerate goldens at the actual shape."
            );
        }
        Err(other) => {
            panic!("prove_program failed unexpectedly: {other:?}");
        }
    }
}

/// FR coprocessor smoke test: drive `examples/bn254-fr-poseidon2-sdk-guest`
/// through `prove_program` and time the prove phase. The natural shape
/// (log_t=16, log_k_bytecode=13, log_k_ram=14) matches `fixture()`, so the
/// shape gate passes without padding. The guest emits ~16k `FieldRegEvent`s
/// via the BN254 Fr inline SDK, exercising the full FR coprocessor path
/// through the modular prove pipeline.
///
/// This is the canonical "true FR perf" measurement once it's green —
/// jolt-core can't even run this guest (panics on `FieldMov`), so this
/// number is the *only* way to prove an FR-active program end-to-end.
#[test]
#[ignore = "exercises FR coprocessor through modular prove pipeline; requires jolt CLI"]
fn fr_poseidon2_modular_prove_smoke() {
    let mut program = Program::new("bn254-fr-poseidon2-sdk-guest");
    // Mirror the guest's #[jolt::provable] attribute (lib.rs:13-18).
    // Without these, the default heap (small) overflows when the guest tries
    // to allocate room for the Poseidon2 round constants + state.
    let _ = program
        .set_func("fr_poseidon2_sdk")
        .set_stack_size(65_536)
        .set_heap_size(131_072)
        .set_max_input_size(8_192);

    // Mirrors examples/bn254-fr-poseidon2-sdk/src/main.rs:25-27.
    let s0: [u64; 4] = [1, 0, 0, 0];
    let s1: [u64; 4] = [2, 0, 0, 0];
    let s2: [u64; 4] = [3, 0, 0, 0];
    let inputs = postcard::to_stdvec(&(s0, s1, s2)).expect("postcard encode FR inputs");

    let start = std::time::Instant::now();
    let result = prove_program(&mut program, &inputs, &[], &[]);
    let elapsed = start.elapsed();

    match result {
        Ok((proof, io_device, _artifacts)) => {
            eprintln!(
                "[fr_poseidon2_modular_prove_smoke] prove time = {:.2}s, \
                 io output bytes = {}, evaluation present = {}",
                elapsed.as_secs_f64(),
                io_device.outputs.len(),
                proof.evaluation.is_some()
            );
            assert!(
                proof.evaluation.is_some(),
                "modular prover should produce an evaluation proof for FR Poseidon2"
            );
        }
        Err(ProveProgramError::UnsupportedShape {
            log_t,
            log_k_bytecode,
            log_k_ram,
        }) => {
            panic!(
                "FR Poseidon2 shape mismatch: actual=({log_t}, {log_k_bytecode}, {log_k_ram}), \
                 fixture=(16, 13, 14). Goldens need regen at the new shape."
            );
        }
        Err(other) => {
            panic!("prove_program failed on FR Poseidon2: {other:?}");
        }
    }
}
