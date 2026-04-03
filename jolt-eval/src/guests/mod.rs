use ark_bn254::Fr;
use jolt_core::curve::Bn254Curve;
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::transcripts::Blake2bTranscript;

pub use jolt_core::guest::program::Program as GuestProgram;
pub use jolt_core::utils::errors::ProofVerifyError;
pub use tracer::JoltDevice;

pub type F = Fr;
pub type C = Bn254Curve;
pub type PCS = DoryCommitmentScheme;
pub type FS = Blake2bTranscript;

pub type Proof = jolt_core::zkvm::proof_serialization::JoltProof<F, C, PCS, FS>;
pub type ProverPreprocessing = jolt_core::zkvm::prover::JoltProverPreprocessing<F, C, PCS>;
pub type VerifierPreprocessing = jolt_core::zkvm::verifier::JoltVerifierPreprocessing<F, C, PCS>;

pub fn prover_preprocessing(
    program: &GuestProgram,
    max_trace_length: usize,
) -> ProverPreprocessing {
    jolt_core::guest::prover::preprocess(program, max_trace_length)
        .expect("prover preprocessing failed")
}

pub fn verifier_preprocessing(prover_pp: &ProverPreprocessing) -> VerifierPreprocessing {
    VerifierPreprocessing::from(prover_pp)
}

pub fn prove(
    program: &GuestProgram,
    prover_pp: &ProverPreprocessing,
    inputs: &[u8],
) -> (Proof, JoltDevice) {
    let mut output_bytes = vec![0u8; program.memory_config.max_output_size as usize];
    let (proof, io_device, _debug) = jolt_core::guest::prover::prove::<F, C, PCS, FS>(
        program,
        inputs,
        &[],
        &[],
        None,
        None,
        &mut output_bytes,
        prover_pp,
    );
    (proof, io_device)
}

pub fn verify(
    verifier_pp: &VerifierPreprocessing,
    proof: Proof,
    io_device: &JoltDevice,
) -> Result<(), ProofVerifyError> {
    jolt_core::guest::verifier::verify::<F, C, PCS, FS>(
        &io_device.inputs,
        None,
        &io_device.outputs,
        proof,
        verifier_pp,
    )
}

/// Verify a proof against claimed (potentially malicious) outputs and panic flag.
pub fn verify_with_claims(
    verifier_pp: &VerifierPreprocessing,
    proof: Proof,
    inputs: &[u8],
    claimed_outputs: &[u8],
    claimed_panic: bool,
) -> Result<(), ProofVerifyError> {
    use common::jolt_device::MemoryConfig;
    use jolt_core::zkvm::verifier::JoltVerifier;

    let memory_layout = &verifier_pp.shared.memory_layout;
    let memory_config = MemoryConfig {
        max_untrusted_advice_size: memory_layout.max_untrusted_advice_size,
        max_trusted_advice_size: memory_layout.max_trusted_advice_size,
        max_input_size: memory_layout.max_input_size,
        max_output_size: memory_layout.max_output_size,
        stack_size: memory_layout.stack_size,
        heap_size: memory_layout.heap_size,
        program_size: Some(memory_layout.program_size),
    };
    let mut io_device = JoltDevice::new(&memory_config);
    io_device.inputs = inputs.to_vec();
    io_device.outputs = claimed_outputs.to_vec();
    io_device.panic = claimed_panic;

    let verifier = JoltVerifier::<F, C, PCS, FS>::new(verifier_pp, proof, io_device, None, None)?;
    verifier.verify()
}
