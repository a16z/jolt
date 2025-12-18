use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_secp256k1_scalar_mul(target_dir);

    let prover_preprocessing = guest::preprocess_prover_secp256k1_scalar_mul(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_secp256k1_scalar_mul(&prover_preprocessing);

    let prove_secp256k1_scalar_mul =
        guest::build_prover_secp256k1_scalar_mul(program, prover_preprocessing);
    let verify_secp256k1_scalar_mul =
        guest::build_verifier_secp256k1_scalar_mul(verifier_preprocessing);

    // arbitrary secp256k1 point as 8 u64s
    // x and y coordinates in little endian order
    // in montgomery form
    let q = [
        0x84c60f988985bb6d,
        0x3771987a8626ed1b,
        0x7d2d842df22e3972,
        0x68c3e1d401738d23,
        0x7ba86c982b250320,
        0x845453face9978fb,
        0xd480f970fa1501a4,
        0xd9ccbc62a5f896f9,
    ];
    // two arbitrary scalars as elements of Fr in montgomery form
    let u = [
        0x0FEDCBA987654321,
        0x1234567890ABCDEF,
        0x3333333333333333,
        0x4444444444444444,
    ];
    let v = [
        0x5555555555555555,
        0x6666666666666666,
        0x7777777777777777,
        0x8888888888888888,
    ];
    let native_output = guest::secp256k1_scalar_mul(u, v, q);
    let now = Instant::now();
    let (output, proof, program_io) = prove_secp256k1_scalar_mul(u, v, q);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_secp256k1_scalar_mul(u, v, q, output, program_io.panic, proof);

    assert_eq!(output, native_output, "output mismatch");
    info!("output: {:?}", output);
    info!("native_output: {:?}", native_output);
    info!("valid: {is_valid}");
}
