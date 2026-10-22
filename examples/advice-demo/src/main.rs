use jolt_sdk::MemoryBudget;
use std::error::Error;
use std::time::Instant;
use tracing::info;

pub fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    tracing_subscriber::fmt::init();

    // `--memory-budget-bytes N` proves through the budgeted prover.
    let args: Vec<String> = std::env::args().collect();
    let memory_budget = args
        .iter()
        .position(|arg| arg == "--memory-budget-bytes")
        .map(|flag| {
            let bytes = args.get(flag + 1).and_then(|bytes| bytes.parse().ok());
            MemoryBudget::from_bytes(bytes.expect("--memory-budget-bytes takes a byte count"))
        });

    let target_dir = "/tmp/jolt-guest-targets";

    let n = 221u8;
    let a = vec![1usize, 2, 3, 4, 5];
    let b = vec![0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9];

    let mut program = guest::compile_advice_demo(target_dir);
    let shared_preprocessing = guest::preprocess_shared_advice_demo(&mut program).unwrap();
    let prover_preprocessing = guest::preprocess_prover_advice_demo(shared_preprocessing.clone());
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_advice_demo(&prover_preprocessing);
    let verify_advice_demo = guest::build_verifier_advice_demo(verifier_preprocessing);

    let now = Instant::now();
    let (output, proof, program_io) = match memory_budget {
        None => {
            let prove_advice_demo = guest::build_prover_advice_demo(program, prover_preprocessing);
            prove_advice_demo(n, a.clone(), b.clone())
        }
        Some(budget) => {
            let prove_advice_demo = guest::build_budgeted_prover_advice_demo(
                program,
                prover_preprocessing,
                Some(budget),
            );
            prove_advice_demo(n, a.clone(), b.clone())?
        }
    };
    let prove_time = now.elapsed();
    info!("Prover runtime: {} s", prove_time.as_secs_f64());

    let trace_length_advice = proof.trace_length;

    verify_advice_demo(n, a, b, output, program_io.panic, proof)
        .expect("Proof verification failed");

    info!("Trace length: {} cycles", trace_length_advice);
    Ok(())
}
