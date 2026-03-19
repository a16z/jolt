use jolt_inlines_sha2 as _;
use jolt_sdk::{DoryContext, DoryGlobals, DoryLayout, UntrustedAdvice};
use std::time::Instant;
use tracing::info;

fn dory_layout_from_env() -> DoryLayout {
    match std::env::var("JOLT_DORY_LAYOUT")
        .unwrap_or_else(|_| "cycle".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "cycle" | "cyclemajor" => DoryLayout::CycleMajor,
        "address" | "addressmajor" | "addr" => DoryLayout::AddressMajor,
        other => panic!("invalid JOLT_DORY_LAYOUT={other}; expected cycle|address"),
    }
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let bytecode_chunk = std::env::args()
        .skip_while(|arg| arg != "--committed-bytecode")
        .nth(1)
        .map(|arg| arg.parse().unwrap());

    let layout = dory_layout_from_env();
    DoryGlobals::initialize_context(1, 1, DoryContext::Main, Some(layout))
        .expect("failed to initialize Dory layout");
    info!("dory layout: {:?}", layout);

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_sha2_chain(target_dir);

    let (prover_preprocessing, verifier_preprocessing) = if let Some(chunk_count) = bytecode_chunk {
        info!("bytecode_chunk_count: {}", chunk_count);
        let prover_preprocessing =
            guest::preprocess_committed_sha2_chain(&mut program, chunk_count);
        let verifier_preprocessing =
            guest::verifier_preprocessing_from_prover_sha2_chain(&prover_preprocessing);
        (prover_preprocessing, verifier_preprocessing)
    } else {
        let shared_preprocessing = guest::preprocess_shared_sha2_chain(&mut program);
        let prover_preprocessing =
            guest::preprocess_prover_sha2_chain(shared_preprocessing.clone());
        let verifier_preprocessing = guest::preprocess_verifier_sha2_chain(
            shared_preprocessing,
            prover_preprocessing.generators.to_verifier_setup(),
            None,
        );
        (prover_preprocessing, verifier_preprocessing)
    };

    let prove_sha2_chain = guest::build_prover_sha2_chain(program, prover_preprocessing);
    let verify_sha2_chain = guest::build_verifier_sha2_chain(verifier_preprocessing);

    let input = [5u8; 32];
    let iters = 10;

    let native_output = guest::sha2_chain(UntrustedAdvice::new(input), UntrustedAdvice::new(iters));
    let now = Instant::now();
    let (output, proof, program_io) =
        prove_sha2_chain(UntrustedAdvice::new(input), UntrustedAdvice::new(iters));
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_sha2_chain(output, program_io.panic, proof);

    assert_eq!(output, native_output, "output mismatch");
    if !is_valid {
        return Err(std::io::Error::other("verification failed").into());
    }
    info!("output: {}", hex::encode(output));
    info!("valid: {is_valid}");
    Ok(())
}
