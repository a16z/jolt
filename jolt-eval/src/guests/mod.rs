pub mod btreemap;
pub mod fibonacci;
pub mod secp256k1_ecdsa;
pub mod sha2;
pub mod sha2_chain;
pub mod sha3;
pub mod sha3_chain;

pub use jolt_verifier::VerifierError;

use common::constants::{DEFAULT_MAX_TRUSTED_ADVICE_SIZE, DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE};
use common::jolt_device::MemoryConfig;
use jolt_prover::dory::DoryProverPreprocessing;

pub use btreemap::BTreeMapOps;
pub use fibonacci::Fibonacci;
pub use jolt_host::Program as GuestProgram;
pub use secp256k1_ecdsa::Secp256k1EcdsaVerify;
pub use sha2::Sha2;
pub use sha2_chain::Sha2Chain;
pub use sha3::Sha3;
pub use sha3_chain::Sha3Chain;
pub use tracer::JoltDevice;

pub type VerifierField = jolt_field::Fr;
pub type VerifierPCS = jolt_dory::DoryScheme;
pub type VerifierVC = jolt_crypto::Pedersen<jolt_crypto::Bn254G1>;
pub type VerifierTranscript = jolt_transcript::LegacyBlake2bTranscript<VerifierField>;

pub type Proof = jolt_verifier::JoltProof<VerifierPCS, VerifierVC>;
pub type ProverPreprocessing = DoryProverPreprocessing;
pub type VerifierPreprocessing = jolt_verifier::JoltVerifierPreprocessing<VerifierPCS, VerifierVC>;

pub fn prover_preprocessing(
    program: &mut GuestProgram,
    memory_config: MemoryConfig,
    max_trace_length: usize,
) -> ProverPreprocessing {
    jolt_sdk::preprocess_program(program, memory_config, max_trace_length, None)
        .expect("prover preprocessing failed")
}

pub fn verifier_preprocessing(prover_pp: &ProverPreprocessing) -> VerifierPreprocessing {
    prover_pp.verifier_preprocessing()
}

pub fn prove(
    program: &GuestProgram,
    prover_pp: &ProverPreprocessing,
    inputs: &[u8],
) -> (Proof, JoltDevice) {
    jolt_sdk::prove_program(program, prover_pp, inputs, &[], &[], None, None, None)
        .expect("prover should produce verifier-native proof")
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
    )
}

// ── GuestConfig ─────────────────────────────────────────────────────

/// Trait for configuring which guest program to benchmark.
pub trait GuestConfig: Default + Send + Sync {
    /// Cargo package name (e.g. "fibonacci-guest").
    fn package(&self) -> &str;

    /// Objective-neutral guest label including parameters
    /// (e.g. "fibonacci_400000"). Objectives prefix it with their own name.
    fn label(&self) -> String;

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
    fn bench_name(&self) -> String {
        format!("prover_time_{}", self.label())
    }
}
