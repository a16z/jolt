#[cfg(test)]
#[cfg(feature = "host")]
mod tests {
    use jolt_sdk::{JoltVerifierPreprocessing, RV64IMACProof, RV64IMACVerifier, Serializable};

    #[test]
    fn verify_proof() {
        let preprocessing =
            JoltVerifierPreprocessing::read_from_target_dir("tests/fixtures/").unwrap();
        let proof = RV64IMACProof::from_file("tests/fixtures/fib_proof.bin").unwrap();
        let device =
            common::jolt_device::JoltDevice::from_file("tests/fixtures/fib_io_device.bin").unwrap();
        let start = std::time::Instant::now();
        println!("Verifying proof...");
        let verifier = RV64IMACVerifier::new(&preprocessing, proof, device, None, None).unwrap();
        verifier.verify().unwrap();
        let duration = start.elapsed();
        println!("Verification took: {} ms", duration.as_millis());
    }
}
