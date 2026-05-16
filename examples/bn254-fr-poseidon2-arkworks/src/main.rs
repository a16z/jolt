use std::time::Instant;

use guest::{
    compile_fr_poseidon2_arkworks, prove_fr_poseidon2_arkworks, verify_fr_poseidon2_arkworks,
};
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = compile_fr_poseidon2_arkworks(target_dir);

    // Input state (1, 2, 3) — each Fr fits in a single u64 limb.
    let s0: [u64; 4] = [1, 0, 0, 0];
    let s1: [u64; 4] = [2, 0, 0, 0];
    let s2: [u64; 4] = [3, 0, 0, 0];

    let prove_start = Instant::now();
    let (output, bundle) = prove_fr_poseidon2_arkworks(&mut program, s0, s1, s2)
        .expect("modular prove succeeds on fr_poseidon2_arkworks");
    let prove_secs = prove_start.elapsed().as_secs_f64();

    let verify_start = Instant::now();
    let verify_result = verify_fr_poseidon2_arkworks(&bundle, &mut program);
    let verify_secs = verify_start.elapsed().as_secs_f64();
    let valid = verify_result.is_ok();

    info!("=== bn254-fr-poseidon2-arkworks (modular Bolt backend, software Fr) ===");
    info!("prove time : {prove_secs:.3} s");
    info!("verify time: {verify_secs:.3} s");
    info!("output[0]  : {:?}", output[0]);
    info!("output[1]  : {:?}", output[1]);
    info!("output[2]  : {:?}", output[2]);
    info!("valid      : {valid}");

    if let Err(err) = verify_result {
        info!("verify error: {err:?}");
        std::process::exit(1);
    }
}
