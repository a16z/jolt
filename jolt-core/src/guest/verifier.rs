use crate::curve::{Bn254Curve, JoltCurve};
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::commitment_scheme::{StreamingCommitmentScheme, ZkEvalCommitment};
use crate::utils::errors::ProofVerifyError;
use crate::zkvm::verifier::BlindfoldSetup;

use crate::guest::program::Program;
use crate::poly::commitment::dory::DoryCommitmentScheme;
use crate::transcripts::Transcript;
use crate::zkvm::program::ProgramPreprocessing;
use crate::zkvm::proof_serialization::JoltProof;
use crate::zkvm::verifier::JoltSharedPreprocessing;
use crate::zkvm::verifier::JoltVerifier;
use crate::zkvm::verifier::JoltVerifierPreprocessing;
use common::jolt_device::MemoryConfig;
use common::jolt_device::MemoryLayout;
use std::sync::Arc;

pub fn preprocess(
    guest: &Program,
    max_trace_length: usize,
    verifier_setup: <DoryCommitmentScheme as CommitmentScheme>::VerifierSetup,
    blindfold_setup: Option<BlindfoldSetup<Bn254Curve>>,
) -> JoltVerifierPreprocessing<ark_bn254::Fr, Bn254Curve, DoryCommitmentScheme> {
    let (shared, program) = preprocess_shared(guest, max_trace_length);
    JoltVerifierPreprocessing::new_full(shared, verifier_setup, program, blindfold_setup)
}

fn preprocess_shared(
    guest: &Program,
    max_trace_length: usize,
) -> (JoltSharedPreprocessing, Arc<ProgramPreprocessing>) {
    let (bytecode, memory_init, program_size, _e_entry) = guest.decode();

    let mut memory_config = guest.memory_config;
    memory_config.program_size = Some(program_size);
    let memory_layout = MemoryLayout::new(&memory_config);
    let program = Arc::new(ProgramPreprocessing::preprocess(bytecode, memory_init));
    let shared = JoltSharedPreprocessing::new(program.meta(), memory_layout, max_trace_length);
    (shared, program)
}

pub fn verify<
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: StreamingCommitmentScheme<Field = F> + ZkEvalCommitment<C>,
    FS: Transcript,
>(
    inputs_bytes: &[u8],
    trusted_advice_commitment: Option<<PCS as CommitmentScheme>::Commitment>,
    outputs_bytes: &[u8],
    proof: JoltProof<F, C, PCS, FS>,
    preprocessing: &JoltVerifierPreprocessing<F, C, PCS>,
) -> Result<(), ProofVerifyError> {
    use common::jolt_device::JoltDevice;
    let memory_layout = &preprocessing.shared.memory_layout;
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

    io_device.inputs = inputs_bytes.to_vec();
    io_device.outputs = outputs_bytes.to_vec();

    let verifier = JoltVerifier::new(
        preprocessing,
        proof,
        io_device,
        trusted_advice_commitment,
        None,
    )?;
    verifier.verify()
}
