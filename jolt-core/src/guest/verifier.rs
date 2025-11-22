use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::commitment_scheme::StreamingCommitmentScheme;

use crate::guest::program::Program;
use crate::poly::commitment::dory::DoryCommitmentScheme;
use crate::transcripts::Transcript;
use crate::utils::errors::ProofVerifyError;
use crate::zkvm::proof_serialization::JoltProof;
use crate::zkvm::verifier::JoltSharedPreprocessing;
use crate::zkvm::verifier::JoltVerifier;
use crate::zkvm::verifier::JoltVerifierPreprocessing;
use common::jolt_device::MemoryConfig;
use common::jolt_device::MemoryLayout;

pub fn preprocess(
    guest: &Program,
    verifier_setup: <DoryCommitmentScheme as CommitmentScheme>::VerifierSetup,
) -> JoltVerifierPreprocessing<ark_bn254::Fr, DoryCommitmentScheme> {
    let (bytecode, memory_init, program_size) = guest.decode();

    let mut memory_config = guest.memory_config;
    memory_config.program_size = Some(program_size);
    let memory_layout = MemoryLayout::new(&memory_config);
    let shared = JoltSharedPreprocessing::new(bytecode, memory_layout, memory_init);
    JoltVerifierPreprocessing::new(shared, verifier_setup)
}

pub fn verify<F: JoltField, PCS: StreamingCommitmentScheme<Field = F>, FS: Transcript>(
    inputs_bytes: &[u8],
    trusted_advice_commitment: Option<<PCS as CommitmentScheme>::Commitment>,
    outputs_bytes: &[u8],
    proof: JoltProof<F, PCS, FS>,
    preprocessing: &JoltVerifierPreprocessing<F, PCS>,
) -> Result<(), ProofVerifyError> {
    use common::jolt_device::JoltDevice;
    let memory_layout = &preprocessing.shared.memory_layout;
    let memory_config = MemoryConfig {
        max_untrusted_advice_size: memory_layout.max_untrusted_advice_size,
        max_trusted_advice_size: memory_layout.max_trusted_advice_size,
        max_input_size: memory_layout.max_input_size,
        max_output_size: memory_layout.max_output_size,
        stack_size: memory_layout.stack_size,
        memory_size: memory_layout.memory_size,
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
    verifier.verify().unwrap();
    Ok(())
}
