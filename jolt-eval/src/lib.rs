#![allow(non_snake_case)]

// Allow `jolt_eval::` paths in macro-generated code within this crate.
extern crate self as jolt_eval;

pub mod agent;
pub mod guests;
pub mod invariant;
pub mod objective;

use std::collections::HashMap;
use std::sync::Arc;

use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_core::curve::Bn254Curve;
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::transcripts::Blake2bTranscript;

pub use invariant::{
    Invariant, InvariantTargets, InvariantViolation, JoltInvariants, SynthesisTarget,
};
pub use objective::{AbstractObjective, Direction, MeasurementError, Objective};

// Re-exports used by the #[invariant] proc macro generated code.
// Users of the macro don't need to add these to their own Cargo.toml.
pub use arbitrary;
pub use rand;

pub type F = Fr;
pub type C = Bn254Curve;
pub type PCS = DoryCommitmentScheme;
pub type FS = Blake2bTranscript;

pub type Proof = jolt_core::zkvm::proof_serialization::JoltProof<F, C, PCS, FS>;
pub type ProverPreprocessing = jolt_core::zkvm::prover::JoltProverPreprocessing<F, C, PCS>;
pub type VerifierPreprocessing = jolt_core::zkvm::verifier::JoltVerifierPreprocessing<F, C, PCS>;
pub type SharedPreprocessing = jolt_core::zkvm::verifier::JoltSharedPreprocessing;

pub use jolt_core::guest::program::Program as GuestProgram;
pub use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
pub use jolt_core::utils::errors::ProofVerifyError;
pub use jolt_core::zkvm::Serializable;
pub use tracer::JoltDevice;

/// A self-contained test case wrapping a compiled guest program.
///
/// `TestCase` stores the raw ELF bytes and memory configuration so it can
/// reconstruct a `GuestProgram` on demand without requiring `Clone` on the
/// program itself.
pub struct TestCase {
    pub elf_contents: Vec<u8>,
    pub memory_config: common::jolt_device::MemoryConfig,
    pub max_trace_length: usize,
}

impl TestCase {
    pub fn new(program: GuestProgram, max_trace_length: usize) -> Self {
        Self {
            elf_contents: program.elf_contents,
            memory_config: program.memory_config,
            max_trace_length,
        }
    }

    pub fn make_program(&self) -> GuestProgram {
        GuestProgram::new(&self.elf_contents, &self.memory_config)
    }

    /// Create prover preprocessing for this test case.
    pub fn prover_preprocessing(&self) -> ProverPreprocessing {
        let program = self.make_program();
        jolt_core::guest::prover::preprocess(&program, self.max_trace_length)
            .expect("prover preprocessing failed")
    }

    /// Create verifier preprocessing from prover preprocessing.
    pub fn verifier_preprocessing(prover_pp: &ProverPreprocessing) -> VerifierPreprocessing {
        VerifierPreprocessing::from(prover_pp)
    }

    /// Prove execution of this program with the given inputs.
    /// Returns (proof, io_device).
    pub fn prove(&self, prover_pp: &ProverPreprocessing, inputs: &[u8]) -> (Proof, JoltDevice) {
        let program = self.make_program();
        let mut output_bytes = vec![0u8; self.memory_config.max_output_size as usize];
        let (proof, io_device, _debug) = jolt_core::guest::prover::prove::<F, C, PCS, FS>(
            &program,
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

    /// Verify a proof against the given preprocessing and I/O.
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
    ///
    /// Unlike [`verify`], this lets the caller override the output bytes and
    /// panic flag independently, for testing that the verifier rejects
    /// dishonest claims.
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

        let verifier =
            JoltVerifier::<F, C, PCS, FS>::new(verifier_pp, proof, io_device, None, None)?;
        verifier.verify()
    }
}

/// Serialize a proof to bytes.
pub fn serialize_proof(proof: &Proof) -> Vec<u8> {
    let mut buf = Vec::new();
    proof
        .serialize_compressed(&mut buf)
        .expect("proof serialization failed");
    buf
}

/// Deserialize a proof from bytes.
pub fn deserialize_proof(bytes: &[u8]) -> Result<Proof, ark_serialize::SerializationError> {
    Proof::deserialize_compressed(bytes)
}

/// Run all provided invariants, returning results keyed by name.
pub fn check_all_invariants(
    invariants: &[JoltInvariants],
    num_random: usize,
) -> HashMap<String, Vec<Result<(), InvariantViolation>>> {
    invariants
        .iter()
        .map(|inv| {
            let name = inv.name().to_string();
            let results = inv.run_checks(num_random);
            (name, results)
        })
        .collect()
}

/// Measure all provided objectives, returning results keyed by name.
pub fn measure_all_objectives(
    objectives: &[Objective],
) -> HashMap<String, Result<f64, MeasurementError>> {
    objectives
        .iter()
        .map(|obj| {
            let name = obj.name().to_string();
            let result = obj.collect_measurement();
            (name, result)
        })
        .collect()
}

/// Shared setup that can be reused across multiple invariants/objectives
/// operating on the same program.
pub struct SharedSetup {
    pub test_case: Arc<TestCase>,
    pub prover_preprocessing: Arc<ProverPreprocessing>,
    pub verifier_preprocessing: Arc<VerifierPreprocessing>,
}

impl SharedSetup {
    pub fn new(test_case: TestCase) -> Self {
        Self::new_from_arc(Arc::new(test_case))
    }

    pub fn new_from_arc(test_case: Arc<TestCase>) -> Self {
        let prover_pp = test_case.prover_preprocessing();
        let verifier_pp = TestCase::verifier_preprocessing(&prover_pp);
        Self {
            test_case,
            prover_preprocessing: Arc::new(prover_pp),
            verifier_preprocessing: Arc::new(verifier_pp),
        }
    }
}
