#[cfg(test)]
#[cfg(feature = "host")]
mod tests {
    use jolt_sdk::{
        deserialize_verifier_object, JoltDevice, JoltVerifierPreprocessing, RV64IMACProof,
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

        let preprocessing_bytes = std::fs::read(preprocessing_path).unwrap();
        let preprocessing: JoltVerifierPreprocessing =
            deserialize_verifier_object(&preprocessing_bytes).unwrap();
        let proof_bytes = std::fs::read(proof_path).unwrap();
        let proof: RV64IMACProof = deserialize_verifier_object(&proof_bytes).unwrap();
        let io_bytes = std::fs::read(io_path).unwrap();
        let device: JoltDevice = deserialize_verifier_object(&io_bytes).unwrap();
        let start = std::time::Instant::now();
        println!("Verifying proof...");
        let result = jolt_sdk::jolt_verifier::verify::<
            jolt_sdk::VerifierField,
            jolt_sdk::VerifierPCS,
            jolt_sdk::VerifierVC,
            jolt_sdk::VerifierTranscript,
        >(&preprocessing, &device, &proof, None, false);
        let duration = start.elapsed();
        println!("Verification took: {} ms", duration.as_millis());
        assert!(result.is_ok(), "Verifier failed: {:?}", result.err());
    }
}
