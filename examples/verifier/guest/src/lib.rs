use jolt::{end_cycle_tracking, jolt_println, start_cycle_tracking, Jolt, PCS};
use jolt::{JoltRV32IM, JoltVerifierPreprocessing, RV32IMJoltProof, Serializable};

mod fib_io_device_bytes;
mod fib_proof_bytes;
mod jolt_verifier_preprocessing_bytes;

#[jolt::provable(
    memory_size = 33554432,
    max_trace_length = 67108864,
    stack_size = 1048576
)]
fn verifier() {
    start_cycle_tracking("preprocessing");
    let preprocessing: JoltVerifierPreprocessing<ark_bn254::Fr, PCS> =
        JoltVerifierPreprocessing::deserialize_from_bytes_unchecked(
            jolt_verifier_preprocessing_bytes::JOLT_VERIFIER_PREPROCESSING_BYTES,
        )
        .unwrap();
    end_cycle_tracking("preprocessing");
    start_cycle_tracking("proof");
    let proof = RV32IMJoltProof::deserialize_from_bytes_unchecked(fib_proof_bytes::FIB_PROOF_BYTES).unwrap();
    end_cycle_tracking("proof");
    start_cycle_tracking("device");
    let device =
        jolt::JoltDevice::deserialize_from_bytes_unchecked(fib_io_device_bytes::FIB_IO_DEVICE_BYTES).unwrap();
    end_cycle_tracking("device");
    // assert!(device.memory_layout.stack_size > 1024);
    // assert!(proof.trace_length > 0);
    start_cycle_tracking("verification");
    let verifier = JoltRV32IM::verify(&preprocessing, proof, device, None);
    end_cycle_tracking("verification");

    assert!(verifier.is_ok(), "Verifier failed: {:?}", verifier.err());
}
