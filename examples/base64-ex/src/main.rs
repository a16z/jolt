use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_base64_encode(target_dir);

    let prover_preprocessing = guest::preprocess_prover_base64_encode(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_base64_encode(&prover_preprocessing);

    let prove = guest::build_prover_base64_encode(program, prover_preprocessing);
    let verify = guest::build_verifier_base64_encode(verifier_preprocessing);

    let input = b"hello jolt base64!";
    let now = Instant::now();
    let (chunks, proof, program_io) = prove(input);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());

    let is_valid = verify(input, chunks, program_io.panic, proof);

    // merge chunks and trim zero tail
    let (chunk1, chunk2) = chunks;
    let mut out = [0u8; 64];
    out[..32].copy_from_slice(&chunk1);
    out[32..].copy_from_slice(&chunk2);
    let encoded = out.iter().take_while(|&&b| b != 0).copied().collect::<Vec<_>>();
    info!("output: {}", String::from_utf8_lossy(&encoded));
    info!("valid: {is_valid}");
}
