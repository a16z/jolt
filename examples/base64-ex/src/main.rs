use guest::B64Array;
use std::time::Instant;
use tracing::info; // <-- import the new-type

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
    let (B64Array(output), proof, program_io) = prove(input);
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());

    let is_valid = verify(input, B64Array(output), program_io.panic, proof);

    let encoded = output
        .iter()
        .take_while(|&&b| b != 0)
        .copied()
        .collect::<Vec<_>>();
    info!("output: {}", String::from_utf8_lossy(&encoded));
    info!("valid: {is_valid}");
}
