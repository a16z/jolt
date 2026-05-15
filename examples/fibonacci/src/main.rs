use std::time::Instant;

use guest::{compile_fib, prove_fib, verify_fib};
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = compile_fib(target_dir);

    let prove_start = Instant::now();
    let (output, bundle) = prove_fib(&mut program, 50).expect("modular prove succeeds on fib");
    let prove_secs = prove_start.elapsed().as_secs_f64();

    let verify_start = Instant::now();
    let verify_result = verify_fib(&bundle, &mut program);
    let verify_secs = verify_start.elapsed().as_secs_f64();
    let valid = verify_result.is_ok();

    info!("=== fibonacci (modular Bolt backend) ===");
    info!("prove time : {prove_secs:.3} s");
    info!("verify time: {verify_secs:.3} s");
    info!("output     : {output}");
    info!("valid      : {valid}");

    if let Err(err) = verify_result {
        info!("verify error: {err:?}");
        std::process::exit(1);
    }
}
