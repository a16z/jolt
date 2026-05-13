//! Modular prove+verify driver for the inline-FR Poseidon2 example.
//!
//! The guest is annotated with `#[jolt::provable(backend = "modular")]`, so
//! `guest` exposes `compile_fr_poseidon2_sdk`, `prove_fr_poseidon2_sdk`, and
//! `verify_fr_poseidon2_sdk` (macro-generated, all wired to
//! `jolt_host::{prove_program, verify_proof}` under the hood).

use std::time::Instant;

use guest::{compile_fr_poseidon2_sdk, prove_fr_poseidon2_sdk, verify_fr_poseidon2_sdk};
use jolt_profiling::PeakRssSampler;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let s0: [u64; 4] = [1, 0, 0, 0];
    let s1: [u64; 4] = [2, 0, 0, 0];
    let s2: [u64; 4] = [3, 0, 0, 0];

    let mut program = compile_fr_poseidon2_sdk();

    let rss = PeakRssSampler::start().expect("start RSS sampler");
    let prove_start = Instant::now();
    let output = prove_fr_poseidon2_sdk(&mut program, s0, s1, s2)
        .expect("modular prove succeeds on FR Poseidon2");
    let prove_secs = prove_start.elapsed().as_secs_f64();
    let peak_rss_mb = rss.finish();

    let verify_start = Instant::now();
    let verify_result = verify_fr_poseidon2_sdk(&output, &mut program);
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
