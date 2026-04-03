use ark_bn254::Fr;
use jolt_core::curve::Bn254Curve;
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::transcripts::Blake2bTranscript;

use common::constants::{DEFAULT_MAX_TRUSTED_ADVICE_SIZE, DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE};
use common::jolt_device::MemoryConfig;

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

// ── Concrete guest configurations ───────────────────────────────────

/// Fibonacci guest: computes fib(n).
pub struct Fibonacci(pub u32);

impl Default for Fibonacci {
    fn default() -> Self {
        Self(100)
    }
}

impl GuestConfig for Fibonacci {
    fn package(&self) -> &str {
        "fibonacci-guest"
    }
    fn input(&self) -> Vec<u8> {
        postcard::to_stdvec(&self.0).unwrap()
    }
    fn bench_name(&self) -> String {
        format!("prover_time_fibonacci_{}", self.0)
    }
}

/// SHA-2 chain guest: iteratively hashes input `num_iters` times.
pub struct Sha2Chain {
    pub input: [u8; 32],
    pub num_iters: u32,
}

impl Default for Sha2Chain {
    fn default() -> Self {
        Self {
            input: [5u8; 32],
            num_iters: 100,
        }
    }
}

impl GuestConfig for Sha2Chain {
    fn package(&self) -> &str {
        "sha2-chain-guest"
    }
    fn input(&self) -> Vec<u8> {
        postcard::to_stdvec(&(self.input, self.num_iters)).unwrap()
    }
    fn bench_name(&self) -> String {
        format!("prover_time_sha2_chain_{}", self.num_iters)
    }
}

/// Secp256k1 ECDSA signature verification guest.
pub struct Secp256k1EcdsaVerify {
    pub z: [u64; 4],
    pub r: [u64; 4],
    pub s: [u64; 4],
    pub q: [u64; 8],
}

impl Default for Secp256k1EcdsaVerify {
    fn default() -> Self {
        // Test vector from examples/secp256k1-ecdsa-verify: "hello world"
        Self {
            z: [
                0x9088f7ace2efcde9,
                0xc484efe37a5380ee,
                0xa52e52d7da7dabfa,
                0xb94d27b9934d3e08,
            ],
            r: [
                0xb8fc413b4b967ed8,
                0x248d4b0b2829ab00,
                0x587f69296af3cd88,
                0x3a5d6a386e6cf7c0,
            ],
            s: [
                0x66a82f274e3dcafc,
                0x299a02486be40321,
                0x6212d714118f617e,
                0x9d452f63cf91018d,
            ],
            q: [
                0x0012563f32ed0216,
                0xee00716af6a73670,
                0x91fc70e34e00e6c8,
                0xeeb6be8b9e68868b,
                0x4780de3d5fda972d,
                0xcb1b42d72491e47f,
                0xdc7f31262e4ba2b7,
                0xdc7b004d3bb2800d,
            ],
        }
    }
}

impl GuestConfig for Secp256k1EcdsaVerify {
    fn package(&self) -> &str {
        "secp256k1-ecdsa-verify-guest"
    }
    fn memory_config(&self) -> MemoryConfig {
        MemoryConfig {
            max_input_size: 4096,
            max_output_size: 4096,
            max_untrusted_advice_size: DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE,
            max_trusted_advice_size: DEFAULT_MAX_TRUSTED_ADVICE_SIZE,
            stack_size: 4096,
            heap_size: 100000,
            program_size: None,
        }
    }
    fn input(&self) -> Vec<u8> {
        postcard::to_stdvec(&(self.z, self.r, self.s, self.q)).unwrap()
    }
    fn bench_name(&self) -> String {
        "prover_time_secp256k1_ecdsa_verify".to_string()
    }
}
