//! Modular prove+verify driver for the inline-FR Poseidon2 example.
//!
//! jolt-core can't run this guest (it panics on `FieldMov`), so we drive
//! the modular Bolt-based stack via `jolt_host::prove_program` and then
//! round-trip through `jolt_host::verify_proof`. Reports prove time,
//! verify time, peak RSS, and proof validity.

use std::time::Instant;

use jolt_host::{prove_program, verify_proof};
use jolt_profiling::PeakRssSampler;
use jolt_trace::Program;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let mut program = Program::new("bn254-fr-poseidon2-sdk-guest");
    // Mirror the guest's `#[jolt::provable]` attribute. Without these the
    // default heap (small) overflows when the guest allocates the Poseidon2
    // round constants + state.
    let _ = program
        .set_func("fr_poseidon2_sdk")
        .set_stack_size(65_536)
        .set_heap_size(131_072)
        .set_max_input_size(8_192);

    let s0: [u64; 4] = [1, 0, 0, 0];
    let s1: [u64; 4] = [2, 0, 0, 0];
    let s2: [u64; 4] = [3, 0, 0, 0];
    let inputs = postcard::to_stdvec(&(s0, s1, s2)).expect("postcard encode FR inputs");

    let rss = PeakRssSampler::start().expect("start RSS sampler");
    let prove_start = Instant::now();
    let output = prove_program(&mut program, &inputs, &[], &[])
        .expect("modular prove_program succeeds on FR Poseidon2");
    let prove_secs = prove_start.elapsed().as_secs_f64();
    let peak_rss_mb = rss.finish();

    let verify_start = Instant::now();
    let verify_result = verify_proof(&output, &mut program);
    let verify_secs = verify_start.elapsed().as_secs_f64();
    let valid = verify_result.is_ok();

    info!("=== bn254-fr-poseidon2-sdk (modular Bolt + FR coprocessor) ===");
    info!("prove time : {prove_secs:.3} s");
    info!("verify time: {verify_secs:.3} s");
    info!("peak RSS   : {peak_rss_mb} MB");
    info!("valid: {valid}");

    if let Err(err) = verify_result {
        info!("verify error: {err:?}");
        std::process::exit(1);
    }
}
