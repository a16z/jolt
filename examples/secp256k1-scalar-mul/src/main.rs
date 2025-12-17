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

    // generator point
    let point = jolt_inlines_secp256k1::Secp256k1Point::generator().to_u64_arr();
    // arbitrary scalar
    let scalar = [
        0x0FEDCBA987654321,
        0x123456789ABCDEF0,
        0x3333333333333333,
        0x4444444444444444,
    ];
    let native_output = guest::secp256k1_scalar_mul(scalar, point);
    let now = Instant::now();
    let (output, proof, program_io) = prove_secp256k1_scalar_mul(scalar, point);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_secp256k1_scalar_mul(scalar, point, output, program_io.panic, proof);

    assert_eq!(output, native_output, "output mismatch");
    info!("output: {:?}", output);
    info!("native_output: {:?}", native_output);
    info!("valid: {is_valid}");
}
