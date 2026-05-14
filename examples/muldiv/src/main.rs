use std::time::Instant;

use guest::{compile_muldiv, prove_muldiv, verify_muldiv};
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let mut program = compile_muldiv();

    let prove_start = Instant::now();
    let (output, bundle) = prove_muldiv(&mut program, 12031293, 17, 92)
        .expect("modular prove succeeds on muldiv");
    let prove_secs = prove_start.elapsed().as_secs_f64();

    let verify_start = Instant::now();
    let verify_result = verify_muldiv(&bundle, &mut program);
    let verify_secs = verify_start.elapsed().as_secs_f64();
    let valid = verify_result.is_ok();

    info!("=== muldiv (modular Bolt backend) ===");
    info!("prove time : {prove_secs:.3} s");
    info!("verify time: {verify_secs:.3} s");
    info!("output     : {output}");
    info!("valid      : {valid}");

    if let Err(err) = verify_result {
        info!("verify error: {err:?}");
        std::process::exit(1);
    }
}
