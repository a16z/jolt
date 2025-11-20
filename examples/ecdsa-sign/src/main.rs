use sha2::{Digest, Sha256};
use std::time::Instant;
use tracing::info;

// Example private key (32 bytes) - DO NOT use in production!
const PRIVATE_KEY: [u8; 32] = [
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
    0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20,
];

// Example message to sign
const MESSAGE: &[u8] = b"Hello, ECDSA!";

pub fn main() {
    tracing_subscriber::fmt::init();

    info!("ECDSA Signing Example (BN254 Signature)");
    info!("================================");

    // Hash the message
    let mut hasher = Sha256::new();
    hasher.update(MESSAGE);
    let message_hash: [u8; 32] = hasher.finalize().into();

    info!("Message: {:?}", std::str::from_utf8(MESSAGE).unwrap());
    info!("Message hash: 0x{}", hex::encode(&message_hash));

    // Compile and prepare for proving
    info!("\nCompiling guest program...");
    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_ecdsa_sign(target_dir);

    info!("Generating prover preprocessing...");
    let prover_preprocessing = guest::preprocess_prover_ecdsa_sign(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_ecdsa_sign(&prover_preprocessing);

    let prove_ecdsa_sign = guest::build_prover_ecdsa_sign(program, prover_preprocessing);

    // Analyze the program
    let program_summary = guest::analyze_ecdsa_sign(PRIVATE_KEY, message_hash);
    let trace_length = program_summary.trace.len();
    let max_trace_length = if trace_length == 0 {
        1
    } else {
        (trace_length - 1).next_power_of_two()
    };
    info!("\nProgram Analysis:");
    info!("Trace length: {}", trace_length);
    info!("Max trace length: {}", max_trace_length);

    // Generate proof
    info!("\nGenerating proof...");
    let now = Instant::now();
    let ((r, s), _proof, _io_device) = prove_ecdsa_sign(PRIVATE_KEY, message_hash);
    let proving_time = now.elapsed();
    info!("Prover runtime: {:.3} s", proving_time.as_secs_f64());

    info!("\nSignature (BN254 ECDSA):");
    info!("  r: 0x{}", hex::encode(r));
    info!("  s: 0x{}", hex::encode(s));
}
