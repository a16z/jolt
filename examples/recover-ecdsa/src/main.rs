use guest::recover;
use jolt_sdk::serialize_and_print_size;
use secp256k1::{Message, PublicKey, Secp256k1, SecretKey};
use std::time::Instant;
use tracing::info;

const SECRET_KEY: [u8; 32] = [
    59, 148, 11, 85, 134, 130, 61, 253, 2, 174, 59, 70, 27, 180, 51, 107, 94, 203, 174, 253, 102,
    39, 170, 146, 46, 252, 4, 143, 236, 12, 136, 28,
];

pub fn main() {
    tracing_subscriber::fmt::init();

    let secp = Secp256k1::new();

    let seckey = SecretKey::from_slice(&SECRET_KEY).unwrap();
    let _pubkey = PublicKey::from_secret_key(&secp, &seckey);
    let msg_digest = *b"this must be secure hash output.";

    let signature = secp.sign_ecdsa_recoverable(&Message::from_digest(msg_digest), &seckey);
    let (recovery_id, sig_bytes_array) = signature.serialize_compact();

    let sig_bytes = [&sig_bytes_array[..], &[recovery_id as u8]].concat();
    assert!(sig_bytes.len() == 65);

    let _ = recover(&sig_bytes, msg_digest);

    let save_to_disk = std::env::args().any(|arg| arg == "--save");

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_recover(target_dir);

    let shared_preprocessing = guest::preprocess_shared_recover(&mut program);
    let prover_preprocessing = guest::preprocess_prover_recover(shared_preprocessing.clone());
    let verifier_preprocessing = guest::preprocess_verifier_recover(
        shared_preprocessing,
        prover_preprocessing.generators.to_verifier_setup(),
    );

    if save_to_disk {
        serialize_and_print_size(
            "Verifier Preprocessing",
            "/tmp/jolt_verifier_preprocessing.dat",
            &verifier_preprocessing,
        )
        .expect("Could not serialize preprocessing.");
    }

    let prove_recover = guest::build_prover_recover(program, prover_preprocessing);
    let verify_recover = guest::build_verifier_recover(verifier_preprocessing);

    let program_summary = guest::analyze_recover(&sig_bytes, msg_digest);
    let trace_length = program_summary.trace.len();
    let max_trace_length = if trace_length == 0 {
        1
    } else {
        (trace_length - 1).next_power_of_two()
    };
    info!("Trace length: {trace_length:?}");
    info!("Max trace length: {max_trace_length:?}");

    let now = Instant::now();
    let (output, proof, io_device) = prove_recover(&sig_bytes, msg_digest);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());

    let is_valid = verify_recover(&sig_bytes, msg_digest, output, io_device.panic, proof);
    info!("output: {output}");
    info!("valid: {is_valid}");
}
