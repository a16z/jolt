use jolt::{end_cycle_tracking, start_cycle_tracking};
use jolt_verifier::zkvm::{JoltRV32IM, JoltVerifierPreprocessing, RV32IMJoltProof, Serializable};
use jolt_verifier::{common, DoryCommitmentScheme, Jolt};

mod fib_io_device_bytes;
mod fib_proof_bytes;
mod jolt_verifier_preprocessing;

#[jolt::provable(
    memory_size = 10485760,
    max_trace_length = 67108864,
    stack_size = 65536
)]
fn failing_verifier() {
    start_cycle_tracking("preprocessing");
    let preprocessing: JoltVerifierPreprocessing<ark_bn254::Fr, DoryCommitmentScheme> =
        JoltVerifierPreprocessing::deserialize_from_bytes(
            jolt_verifier_preprocessing::VERIFIER_PREPROCESSING,
        )
        .unwrap();
    end_cycle_tracking("preprocessing");
    start_cycle_tracking("proof");
    let proof = RV32IMJoltProof::deserialize_from_bytes(fib_proof_bytes::FIB_PROOF_BIN).unwrap();
    end_cycle_tracking("proof");
    start_cycle_tracking("device");
    let device =
        common::jolt_device::JoltDevice::deserialize_from_bytes(fib_io_device_bytes::FIB_IO_DEVICE)
            .unwrap();
    end_cycle_tracking("device");
    println!(
        "Bytecode length: {:?}",
        preprocessing.shared.bytecode.bytecode.len()
    );
    // assert!(device.memory_layout.stack_size > 1024);
    // assert!(proof.trace_length > 0);
    start_cycle_tracking("verification");
    let verifier = JoltRV32IM::verify(&preprocessing, proof, device, None);
    end_cycle_tracking("verification");
    assert!(verifier.is_ok(), "Verifier failed: {:?}", verifier.err());
}
