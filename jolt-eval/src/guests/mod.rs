pub mod fibonacci;
pub mod secp256k1_ecdsa;
pub mod sha2_chain;

use ark_bn254::Fr;
use jolt_prover_legacy::curve::Bn254Curve;
use jolt_prover_legacy::poly::commitment::dory::DoryCommitmentScheme;
use jolt_prover_legacy::transcripts::Blake2bTranscript;
use jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover;
pub use jolt_verifier::VerifierError;

use common::constants::{DEFAULT_MAX_TRUSTED_ADVICE_SIZE, DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE};
use common::jolt_device::MemoryConfig;

pub use fibonacci::Fibonacci;
pub use jolt_prover_legacy::guest::program::Program as GuestProgram;
pub use secp256k1_ecdsa::Secp256k1EcdsaVerify;
pub use sha2_chain::Sha2Chain;
pub use tracer::JoltDevice;

pub type F = Fr;
pub type C = Bn254Curve;
pub type PCS = DoryCommitmentScheme;
pub type FS = Blake2bTranscript;
pub type VerifierField = jolt_field::Fr;
pub type VerifierPCS = jolt_dory::DoryScheme;
pub type VerifierVC = jolt_crypto::Pedersen<jolt_crypto::Bn254G1>;
pub type VerifierTranscript = jolt_transcript::LegacyBlake2bTranscript<VerifierField>;

pub type Proof = jolt_verifier::JoltProof<VerifierPCS, VerifierVC>;
pub type ProverPreprocessing = jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing<F, C, PCS>;
pub type VerifierPreprocessing = jolt_verifier::JoltVerifierPreprocessing<VerifierPCS, VerifierVC>;

pub fn prover_preprocessing(
    program: &GuestProgram,
    max_trace_length: usize,
) -> ProverPreprocessing {
    jolt_prover_legacy::guest::prover::preprocess(program, max_trace_length)
        .expect("prover preprocessing failed")
}

pub fn verifier_preprocessing(prover_pp: &ProverPreprocessing) -> VerifierPreprocessing {
    verifier_preprocessing_from_prover(prover_pp)
}

pub fn prove(
    program: &GuestProgram,
    prover_pp: &ProverPreprocessing,
    inputs: &[u8],
) -> (Proof, JoltDevice) {
    let mut output_bytes = vec![0u8; program.memory_config.max_output_size as usize];
    let (proof, io_device, _debug) = jolt_prover_legacy::guest::prover::prove::<F, C, PCS, FS>(
        program,
        inputs,
        &[],
        &[],
        None,
        None,
        &mut output_bytes,
        prover_pp,
    )
    .expect("prover should produce verifier-native proof");
    (proof, io_device)
}

pub fn verify(
    verifier_pp: &VerifierPreprocessing,
    proof: Proof,
    io_device: &JoltDevice,
) -> Result<(), VerifierError> {
    jolt_verifier::verify::<VerifierField, VerifierPCS, VerifierVC, VerifierTranscript>(
        verifier_pp,
        io_device,
        &proof,
        None,
        false,
    )
}

/// Verify a proof against claimed (potentially malicious) outputs and panic flag.
pub fn verify_with_claims(
    verifier_pp: &VerifierPreprocessing,
    proof: Proof,
    inputs: &[u8],
    claimed_outputs: &[u8],
    claimed_panic: bool,
) -> Result<(), VerifierError> {
    let memory_layout = verifier_pp.program.memory_layout();
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

    jolt_verifier::verify::<VerifierField, VerifierPCS, VerifierVC, VerifierTranscript>(
        verifier_pp,
        &io_device,
        &proof,
        None,
        false,
    )
}

// ── GuestConfig ─────────────────────────────────────────────────────

/// Trait for configuring which guest program to benchmark.
pub trait GuestConfig: Default + Send + Sync {
    /// Cargo package name (e.g. "fibonacci-guest").
    fn package(&self) -> &str;

    fn memory_config(&self) -> MemoryConfig {
        MemoryConfig {
            max_input_size: 4096,
            max_output_size: 4096,
            max_untrusted_advice_size: DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE,
            max_trusted_advice_size: DEFAULT_MAX_TRUSTED_ADVICE_SIZE,
            stack_size: 4096,
            heap_size: 32768,
            program_size: None,
        }
    }

    /// Serialized program input (postcard-encoded).
    fn input(&self) -> Vec<u8>;

    /// Display name for the benchmark.
    fn bench_name(&self) -> String;
}
