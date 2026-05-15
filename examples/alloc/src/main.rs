use std::time::Instant;

use guest::{compile_alloc, prove_alloc, verify_alloc};
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = compile_alloc(target_dir);

    let input: u32 = 41;
    let prove_start = Instant::now();
    let (output, bundle) =
        prove_alloc(&mut program, input).expect("modular prove succeeds on alloc");
    let prove_secs = prove_start.elapsed().as_secs_f64();

    let verify_start = Instant::now();
    let verify_result = verify_alloc(&bundle, &mut program);
    let verify_secs = verify_start.elapsed().as_secs_f64();
    let valid = verify_result.is_ok();

    info!("=== alloc (modular Bolt backend) ===");
    info!("prove time : {prove_secs:.3} s");
    info!("verify time: {verify_secs:.3} s");
    info!("output     : {output}");
    info!("valid      : {valid}");

    if let Err(err) = verify_result {
        info!("verify error: {err:?}");
        std::process::exit(1);
    }
}
