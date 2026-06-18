#[cfg(test)]
#[cfg(feature = "host")]
mod tests {
    use jolt_sdk::{
        verifier_preprocessing_from_core, verifier_proof_from_core, verify_rv64imac,
        CoreJoltVerifierPreprocessing, CoreRV64IMACProof, Serializable,
    };

    #[test]
    fn verify_proof() {
        let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/");
        let preprocessing_path = format!("{fixtures}/jolt_verifier_preprocessing.dat");
        let proof_path = format!("{fixtures}/fib_proof.bin");
        let io_path = format!("{fixtures}/fib_io_device.bin");
        if !std::path::Path::new(&preprocessing_path).exists()
            || !std::path::Path::new(&proof_path).exists()
            || !std::path::Path::new(&io_path).exists()
        {
            eprintln!("skipping verifier fixture test; run jolt-sdk/tests/gen-fixtures.sh first");
            return;
        }

        let core_preprocessing =
            CoreJoltVerifierPreprocessing::read_from_target_dir(fixtures).unwrap();
        let preprocessing = verifier_preprocessing_from_core(&core_preprocessing);
        let core_proof = CoreRV64IMACProof::from_file(proof_path).unwrap();
        let proof = verifier_proof_from_core(core_proof).unwrap();
        let device = common::jolt_device::JoltDevice::from_file(io_path).unwrap();
        let start = std::time::Instant::now();
        println!("Verifying proof...");
        let result = verify_rv64imac(&preprocessing, &device, &proof, None, false);
        let duration = start.elapsed();
        println!("Verification took: {} ms", duration.as_millis());
        assert!(result.is_ok(), "Verifier failed: {:?}", result.err());
    }
}
