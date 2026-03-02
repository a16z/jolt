use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_secp256k1_ecdsa_verify(target_dir);

    let shared_preprocessing = guest::preprocess_shared_secp256k1_ecdsa_verify(&mut program);
    let prover_preprocessing =
        guest::preprocess_prover_secp256k1_ecdsa_verify(shared_preprocessing.clone());
    let verifier_setup = prover_preprocessing.generators.to_verifier_setup();
    let verifier_preprocessing =
        guest::preprocess_verifier_secp256k1_ecdsa_verify(shared_preprocessing, verifier_setup);

    let prove_secp256k1_ecdsa_verify =
        guest::build_prover_secp256k1_ecdsa_verify(program, prover_preprocessing);
    let verify_secp256k1_ecdsa_verify =
        guest::build_verifier_secp256k1_ecdsa_verify(verifier_preprocessing);

    // custom ECDSA signature test vectors, all as little-endian u64 arrays
    // message hash z (the hash of "hello world")
    let z = [
        0x9088f7ace2efcde9,
        0xc484efe37a5380ee,
        0xa52e52d7da7dabfa,
        0xb94d27b9934d3e08,
    ];
    // signature (r, s)
    let r = [
        0xb8fc413b4b967ed8,
        0x248d4b0b2829ab00,
        0x587f69296af3cd88,
        0x3a5d6a386e6cf7c0,
    ];
    let s = [
        0x66a82f274e3dcafc,
        0x299a02486be40321,
        0x6212d714118f617e,
        0x9d452f63cf91018d,
    ];
    // public key Q (as an uncompressed point)
    let q = [
        0x0012563f32ed0216,
        0xee00716af6a73670,
        0x91fc70e34e00e6c8,
        0xeeb6be8b9e68868b,
        0x4780de3d5fda972d,
        0xcb1b42d72491e47f,
        0xdc7f31262e4ba2b7,
        0xdc7b004d3bb2800d,
    ];
    let now = Instant::now();
    let (output, proof, program_io) = prove_secp256k1_ecdsa_verify(z, r, s, q);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_secp256k1_ecdsa_verify(z, r, s, q, output, program_io.panic, proof);

    info!("output: {:?}", output);
    info!("valid: {is_valid}");
}
