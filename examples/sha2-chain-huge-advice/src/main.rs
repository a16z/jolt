use jolt_sdk::{DoryContext, DoryGlobals, DoryLayout, UntrustedAdvice};
use std::time::Instant;
use tracing::info;

const N: u32 = 1;
const ADVICE_BYTES: usize = 8388608;

fn serialized_advice_size(payload_len: usize) -> usize {
    let payload = vec![7u8; payload_len];
    jolt_sdk::postcard::to_stdvec(&UntrustedAdvice::new(payload.as_slice()))
        .expect("failed to serialize advice input")
        .len()
}

pub fn main() {
    tracing_subscriber::fmt::init();
    let bytecode_chunk = std::env::args()
        .skip_while(|arg| arg != "--committed-bytecode")
        .nth(1)
        .map(|arg| arg.parse().unwrap());
    DoryGlobals::initialize_context(1, 1, DoryContext::Main, Some(DoryLayout::AddressMajor))
        .expect("failed to set Dory layout");

    let advice_bytes = ADVICE_BYTES;
    let serialized_advice_bytes = serialized_advice_size(advice_bytes);
    let max_untrusted_advice_bytes = serialized_advice_bytes.next_power_of_two();

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_fib_huge_advice(target_dir);
    program.set_max_untrusted_advice_size(max_untrusted_advice_bytes as u64);

    let (prover_preprocessing, verifier_preprocessing) = if let Some(chunk_count) = bytecode_chunk {
        let prover_preprocessing =
            guest::preprocess_committed_fib_huge_advice(&mut program, chunk_count);
        let verifier_preprocessing =
            guest::verifier_preprocessing_from_prover_fib_huge_advice(&prover_preprocessing);
        (prover_preprocessing, verifier_preprocessing)
    } else {
        let shared_preprocessing = guest::preprocess_shared_fib_huge_advice(&mut program);
        let prover_preprocessing =
            guest::preprocess_prover_fib_huge_advice(shared_preprocessing.clone());
        let verifier_preprocessing = guest::preprocess_verifier_fib_huge_advice(
            shared_preprocessing,
            prover_preprocessing.generators.to_verifier_setup(),
            None,
        );
        (prover_preprocessing, verifier_preprocessing)
    };

    let prove_fib_huge_advice = guest::build_prover_fib_huge_advice(program, prover_preprocessing);
    let verify_fib_huge_advice = guest::build_verifier_fib_huge_advice(verifier_preprocessing);

    let analysis = guest::analyze_fib_huge_advice(N, UntrustedAdvice::new(&[7u8; 2][..]));
    let execution_trace_length = analysis.trace_len();
    let padded_trace_length = execution_trace_length.next_power_of_two();

    let huge_advice = vec![7u8; advice_bytes];
    let advice_input = UntrustedAdvice::new(huge_advice.as_slice());
    let native_output = guest::fib_huge_advice(N, advice_input);

    let now = Instant::now();
    let (output, proof, program_io) = prove_fib_huge_advice(N, advice_input);
    let trace_length = proof.trace_length;
    info!("Prover runtime: {} s", now.elapsed().as_secs_f64());
    let is_valid = verify_fib_huge_advice(N, output, program_io.panic, proof);

    info!("output: {output}");
    info!("native_output: {native_output}");
    info!(
        "execution trace length: {} and padded trace length: {}",
        execution_trace_length, padded_trace_length
    );
    info!("padded proof trace length: {}", trace_length);
    info!("advice payload bytes: {}", advice_bytes);
    info!("serialized advice bytes: {}", serialized_advice_bytes);
    info!(
        "configured max_untrusted_advice bytes: {}",
        max_untrusted_advice_bytes
    );
    info!(
        "advice_bytes / padded_trace_length = {:.2}",
        advice_bytes as f64 / trace_length as f64
    );
    info!("valid: {is_valid}");

    assert_eq!(output, native_output, "output mismatch");
    // assert_eq!(
    //     trace_length, padded_trace_length,
    //     "analysis and proof trace lengths diverged"
    // );
    // assert_eq!(
    //     advice_bytes, ADVICE_BYTES,
    //     "advice length must match the fixed target"
    // );
    // assert!(
    //     serialized_advice_bytes <= max_untrusted_advice_bytes,
    //     "serialized advice exceeds configured max_untrusted_advice_size"
    // );
    assert!(is_valid, "proof verification failed");
}
