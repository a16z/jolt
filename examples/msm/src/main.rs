use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let seed = 0x12345678u64;

    // Run once natively to verify correctness.
    let native_result = guest::msm_bench(seed);
    info!("Native result: {:?}", native_result);

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_msm_bench(target_dir);

    // Analyze to get cycle counts without proving.
    let program_summary = guest::analyze_msm_bench(seed);
    let trace_length = program_summary.trace.len();

    info!("===========================================");
    info!("MSM Benchmark Results");
    info!("===========================================");
    info!("Total trace length: {}", trace_length);

    // Print instruction breakdown.
    info!("Instruction breakdown:");
    let analysis = program_summary.analyze::<ark_bn254::Fr>();
    for (instr, count) in analysis.iter().take(20) {
        info!("  {}: {}", instr, count);
    }
    info!("===========================================");

    // Optional: run full prove/verify cycle.
    let run_proof = std::env::args().any(|arg| arg == "--prove");
    if run_proof {
        info!("Running full prove/verify cycle...");

        let shared_preprocessing = guest::preprocess_shared_msm_bench(&mut program);
        let prover_preprocessing = guest::preprocess_prover_msm_bench(shared_preprocessing.clone());
        let verifier_preprocessing = guest::preprocess_verifier_msm_bench(
            shared_preprocessing,
            prover_preprocessing.generators.to_verifier_setup(),
        );

        let prove = guest::build_prover_msm_bench(program, prover_preprocessing);
        let verify = guest::build_verifier_msm_bench(verifier_preprocessing);

        let now = std::time::Instant::now();
        let (output, proof, io_device) = prove(seed);
        info!("Prover runtime: {:.2} s", now.elapsed().as_secs_f64());

        let is_valid = verify(seed, output, io_device.panic, proof);
        info!("Output: {:?}", output);
        info!("Proof valid: {}", is_valid);
    }
}
