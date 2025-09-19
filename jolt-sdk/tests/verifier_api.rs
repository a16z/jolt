#[cfg(test)]
#[cfg(feature = "host")]
mod tests {
    use jolt_sdk::{Jolt, JoltRV32IM, JoltVerifierPreprocessing, RV32IMJoltProof, Serializable};

    #[test]
    fn verify_proof() {
        let preprocessing =
            JoltVerifierPreprocessing::read_from_target_dir("tests/fixtures/").unwrap();
        let proof = RV32IMJoltProof::from_file("tests/fixtures/fib_proof.bin").unwrap();
        let device =
            common::jolt_device::JoltDevice::from_file("tests/fixtures/fib_io_device.bin").unwrap();
        let start = std::time::Instant::now();
        println!("Verifying proof...");
        let verifier = JoltRV32IM::verify(&preprocessing, proof, device, None);
        let duration = start.elapsed();
        println!("Verification took: {} ms", duration.as_millis());
        assert!(verifier.is_ok(), "Verifier failed: {:?}", verifier.err());
    }
}
