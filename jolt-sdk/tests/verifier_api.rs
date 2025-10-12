#[cfg(test)]
#[cfg(feature = "host")]
mod tests {
    use jolt_sdk::{
        Jolt, JoltRV64IMAC, JoltVerifierPreprocessing, RV64IMACJoltProof, Serializable,
    };

    #[test]
    fn verify_proof() {
        let preprocessing =
            JoltVerifierPreprocessing::read_from_target_dir("tests/fixtures/").unwrap();
        let proof = RV64IMACJoltProof::from_file("tests/fixtures/fib_proof.bin").unwrap();
        let device =
            common::jolt_device::JoltDevice::from_file("tests/fixtures/fib_io_device.bin").unwrap();
        let start = std::time::Instant::now();
        println!("Verifying proof...");
        let verifier = JoltRV64IMAC::verify(&preprocessing, proof, device, None, None);
        let duration = start.elapsed();
        println!("Verification took: {} ms", duration.as_millis());
        assert!(verifier.is_ok(), "Verifier failed: {:?}", verifier.err());
    }
}
