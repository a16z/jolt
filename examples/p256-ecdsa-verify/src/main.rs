use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_p256_ecdsa_verify(target_dir);

    let shared_preprocessing = guest::preprocess_shared_p256_ecdsa_verify(&mut program).unwrap();
    let prover_preprocessing =
        guest::preprocess_prover_p256_ecdsa_verify(shared_preprocessing.clone());
    let verifier_setup = prover_preprocessing.generators.to_verifier_setup();
    let verifier_preprocessing =
        guest::preprocess_verifier_p256_ecdsa_verify(shared_preprocessing, verifier_setup, None);

    let prove_p256_ecdsa_verify =
        guest::build_prover_p256_ecdsa_verify(program, prover_preprocessing);
    let verify_p256_ecdsa_verify = guest::build_verifier_p256_ecdsa_verify(verifier_preprocessing);

    // P-256 ECDSA test vector (RFC 6979, private key d = 0xC9AFA9D8...)
    // All values are little-endian u64 limbs.

    // message hash z = SHA-256("sample")
    let z = [
        0x219f7c40307c8edf,
        0x83f30a857ad8f656,
        0x06d6364bd78467c1,
        0x4847be4ac21fe68a,
    ];
    // signature (r, s)
    let r = [
        0x61ba8a2e970ae87c,
        0xf81746f8e6b05ab8,
        0x15ab9e9a0f4fc6c8,
        0x42ed5ba7de86be7d,
    ];
    let s = [
        0xde14a271eb1fb4d6,
        0xbb8079f1b5d7dfc7,
        0x86880d7edb977acd,
        0x81f8aa8845318fbf,
    ];
    // public key Q (uncompressed: Qx || Qy)
    let q = [
        0xe669622e60f29fb6,
        0xc049b8923b61fa6c,
        0xc961eb74c6356d68,
        0x60fed4ba255a9d31,
        0x77a3c294d4462299,
        0xf2f1b20c2d7e9f51,
        0xa41ae9e95628bc64,
        0x7903fe1008b8bc99,
    ];

    let now = Instant::now();
    let (output, proof, program_io) = prove_p256_ecdsa_verify(z, r, s, q);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_p256_ecdsa_verify(z, r, s, q, output, program_io.panic, proof);

    info!("output: {:?}", output);
    info!("valid: {is_valid}");
}
