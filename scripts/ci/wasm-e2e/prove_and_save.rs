use jolt_sdk::jolt_prover::zkvm::proof::verifier_preprocessing_from_prover;
use jolt_sdk::serialize_verifier_object;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_fib(target_dir);

    let shared_preprocessing = guest::preprocess_shared_fib(&mut program).unwrap();
    let prover_preprocessing = guest::preprocess_prover_fib(shared_preprocessing);
    let verifier_preprocessing = verifier_preprocessing_from_prover(&prover_preprocessing);

    let pp_bytes = serialize_verifier_object(&verifier_preprocessing).expect("serialize pp");
    std::fs::write("pp.bin", &pp_bytes).expect("write pp");
    info!("preprocessing: {} bytes", pp_bytes.len());

    let prove_fib = guest::build_prover_fib(program, prover_preprocessing);
    let (output, proof, io_device) = prove_fib(50);
    info!("output: {output}");

    let proof_bytes = serialize_verifier_object(&proof).expect("serialize proof");
    std::fs::write("proof.bin", &proof_bytes).expect("write proof");
    info!("proof: {} bytes", proof_bytes.len());

    let io_bytes = serialize_verifier_object(&io_device).expect("serialize io");
    std::fs::write("io.bin", &io_bytes).expect("write io");
    info!("io: {} bytes", io_bytes.len());
}
