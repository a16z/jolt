mod fib_io_device_bytes;
mod fib_proof_bytes;
mod jolt_verifier_preprocessing_bytes;

#[cfg(test)]
#[cfg(feature = "host")]
mod tests {
    use crate::{fib_io_device_bytes, fib_proof_bytes, jolt_verifier_preprocessing_bytes};
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

    #[test]
    fn verify_proof_from_mod_files() {
        let preprocessing = JoltVerifierPreprocessing::deserialize_from_bytes(
            jolt_verifier_preprocessing_bytes::JOLT_VERIFIER_PREPROCESSING_BYTES,
        )
        .unwrap();
        let proof =
            RV32IMJoltProof::deserialize_from_bytes(fib_proof_bytes::FIB_PROOF_BYTES).unwrap();
        let device = common::jolt_device::JoltDevice::deserialize_from_bytes(
            fib_io_device_bytes::FIB_IO_DEVICE_BYTES,
        )
        .unwrap();
        let start = std::time::Instant::now();
        println!("Verifying proof...");
        let verifier = JoltRV32IM::verify(&preprocessing, proof, device, None);
        let duration = start.elapsed();
        println!("Verification took: {} ms", duration.as_millis());
        assert!(verifier.is_ok(), "Verifier failed: {:?}", verifier.err());
    }
}
