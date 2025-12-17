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

    // 4 arbitrary secp256k1 points, each as 8 u64s
    let points = [
        0xfd7914a271ed2e42,
        0x7fb20973e1035805,
        0x8c2c7e3c55347a2f,
        0xe069d2fb3df133fd,
        0x70e6973fb3b3c61e,
        0xaed7312cd8530080,
        0x390fa40885dbc7f2,
        0x3142c3b27c54160e,
        0x62cdfbc1358ff2e7,
        0x95bce326ef8d07c0,
        0x1a0637809a7c16e3,
        0x0197263b9b73d8fe,
        0x921e6ffa3fe39600,
        0xc1b77824c49ecaa6,
        0x25b5d035fbbdcd93,
        0xd25330b456437bc4,
        0xbfc6759e3ab1d57a,
        0x2e822c47f143f7dc,
        0xf8d88465f162255a,
        0xac8cbfb4707c3ba1,
        0x92b8007c0027e3b6,
        0x3d3a2aaa3b129d3c,
        0xc71a36833e579582,
        0x63fa22b365e65edc,
        0x84c60f988985bb6d,
        0x3771987a8626ed1b,
        0x7d2d842df22e3972,
        0x68c3e1d401738d23,
        0x7ba86c982b250320,
        0x845453face9978fb,
        0xd480f970fa1501a4,
        0xd9ccbc62a5f896f9,
    ];
    // arbitrary scalar
    let scalars = [
        0x1234567890ABCDEF1234567890ABCDEFu128,
        0x0FEDCBA9876543210FEDCBA987654321u128,
        0x11111111111111111111111111111111u128,
        0x22222222222222222222222222222222u128,
    ];
    let native_output = guest::secp256k1_scalar_mul(scalars, points);
    let now = Instant::now();
    let (output, proof, program_io) = prove_secp256k1_scalar_mul(scalars, points);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_secp256k1_scalar_mul(scalars, points, output, program_io.panic, proof);

    assert_eq!(output, native_output, "output mismatch");
    info!("output: {:?}", output);
    info!("native_output: {:?}", native_output);
    info!("valid: {is_valid}");
}
