//! Perf-instrumented driver for the inline-FR Poseidon2 example.
//!
//! Runs through `jolt-core`'s monolithic `RV64IMACProver` (via the
//! `#[jolt::provable]` macro-generated host wrappers). Reports raw and
//! padded cycle counts, prove time, throughput in kHz, and peak RSS.
//!
//! NOTE: jolt-core does NOT support the BN254 Fr coprocessor instructions
//! (`FieldOp`, `FieldMov`, etc.), so it panics on `FieldMov` mid-trace. The
//! arkworks software variant (`bn254-fr-poseidon2-arkworks`) is the only
//! example that runs end-to-end through jolt-core. This binary stays for
//! historical reference; expect a panic on cargo run.

use std::time::Instant;

use jolt_profiling::PeakRssSampler;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_fr_poseidon2_sdk(target_dir);

    let shared_preprocessing = guest::preprocess_shared_fr_poseidon2_sdk(&mut program).unwrap();
    let prover_preprocessing =
        guest::preprocess_prover_fr_poseidon2_sdk(shared_preprocessing.clone());
    let verifier_preprocessing = guest::preprocess_verifier_fr_poseidon2_sdk(
        shared_preprocessing,
        prover_preprocessing.generators.to_verifier_setup(),
        None,
    );

    let prove = guest::build_prover_fr_poseidon2_sdk(program, prover_preprocessing);
    let verify = guest::build_verifier_fr_poseidon2_sdk(verifier_preprocessing);

    let s0: [u64; 4] = [1, 0, 0, 0];
    let s1: [u64; 4] = [2, 0, 0, 0];
    let s2: [u64; 4] = [3, 0, 0, 0];

    let rss = PeakRssSampler::start().expect("start RSS sampler");
    let now = Instant::now();
    let (output, proof, program_io) = prove(s0, s1, s2);
    let prove_secs = now.elapsed().as_secs_f64();
    let peak_rss_mb = rss.finish();
    info!("=== bn254-fr-poseidon2-sdk (jolt-core monolithic + FR inline) ===");
    info!("prove time : {prove_secs:.3} s");
    info!("peak RSS   : {peak_rss_mb} MB");

    let is_valid = verify(s0, s1, s2, output, program_io.panic, proof);

    info!("output[0]: {:?}", output[0]);
    info!("output[1]: {:?}", output[1]);
    info!("output[2]: {:?}", output[2]);
    info!("valid: {is_valid}");
}
