use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    let target_dir = "/tmp/jolt-guest-targets";

    // Test both standard atomics and portable-atomic compatibility
    info!("Testing AtomicU64 compatibility (standard vs portable-atomic)...");
    let mut program = guest::compile_atomic_test_u64(target_dir);

    let prover_preprocessing = guest::preprocess_prover_atomic_test_u64(&mut program);
    let verifier_preprocessing =
        guest::verifier_preprocessing_from_prover_atomic_test_u64(&prover_preprocessing);

    let prove = guest::build_prover_atomic_test_u64(program, prover_preprocessing);
    let verify = guest::build_verifier_atomic_test_u64(verifier_preprocessing);

    let now = Instant::now();
    let (output, proof, program_io) = prove(42u64);
    info!(
        "AtomicU64 compatibility test prover runtime: {} s",
        now.elapsed().as_secs_f64()
    );

    let is_valid = verify(42u64, output, program_io.panic, proof);

    // Output is (std_result, portable_result)
    info!("Standard atomic result: {}", output.0);
    info!("Portable-atomic result: {}", output.1);
    info!("Results match: {}", output.0 == output.1);
    info!("Proof valid: {}", is_valid);

    if is_valid && output.0 == output.1 {
        info!("✓ Both standard and portable-atomic operations work correctly!");
        info!("✓ This confirms compatibility between core::sync::atomic and portable-atomic");
        info!("✓ Projects using portable-atomic can migrate to Jolt without code changes");
        info!("✓ passes=lower-atomic handles both implementations correctly");
    } else if is_valid {
        info!("⚠ Proof valid but results differ - unexpected behavior");
    } else {
        info!("✗ Atomic operations test failed");
    }
}
