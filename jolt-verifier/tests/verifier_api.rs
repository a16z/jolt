use jolt_verifier::vm::rv32im_vm::{JoltHyperKZGProof, RV32IMJoltVM, Serializable};
use jolt_verifier::vm::{JoltVerifier, JoltVerifierPreprocessing};

#[test]
fn verify_proof() {
    let preprocessing = JoltVerifierPreprocessing::read_from_target_dir("tests/fixtures/").unwrap();
    let proof = JoltHyperKZGProof::from_file("tests/fixtures/fib_proof.bin").unwrap();
    let device =
        common::jolt_device::JoltDevice::from_file("tests/fixtures/fib_io_device.bin").unwrap();
    let verifier = RV32IMJoltVM::verify(preprocessing, proof.proof, device);
    assert!(verifier.is_ok(), "Verifier failed: {:?}", verifier.err());
}
