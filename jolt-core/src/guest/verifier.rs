use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::commitment_scheme::StreamingCommitmentScheme;

use crate::guest::program::Program;
use crate::poly::commitment::dory::DoryCommitmentScheme;
use crate::transcripts::Transcript;
use crate::utils::errors::ProofVerifyError;
use crate::zkvm::dag::proof_serialization::JoltProof;
use crate::zkvm::{Jolt, JoltRV64IMAC, JoltVerifierPreprocessing};
use common::jolt_device::MemoryConfig;
use common::jolt_device::MemoryLayout;

pub fn preprocess(
    guest: &Program,
    max_trace_length: usize,
) -> JoltVerifierPreprocessing<ark_bn254::Fr, DoryCommitmentScheme> {
    let (bytecode, memory_init, program_size) = guest.decode();

    let mut memory_config = guest.memory_config;
    memory_config.program_size = Some(program_size);
    let memory_layout = MemoryLayout::new(&memory_config);

    let prover_preprocessing = JoltRV64IMAC::prover_preprocess(
        bytecode.to_vec(),
        memory_layout,
        memory_init.to_vec(),
        max_trace_length,
    );

    JoltVerifierPreprocessing::from(&prover_preprocessing)
}

pub fn verify<F, PCS: StreamingCommitmentScheme<Field = F>, FS>(
    inputs_bytes: &[u8],
    trusted_advice_commitment: Option<<PCS as CommitmentScheme>::Commitment>,
    outputs_bytes: &[u8],
    proof: JoltProof<F, PCS, FS>,
    preprocessing: &JoltVerifierPreprocessing<F, PCS>,
) -> Result<(), ProofVerifyError>
where
    F: JoltField,
    FS: Transcript,
    JoltRV64IMAC: Jolt<F, PCS, FS>,
{
    use common::jolt_device::JoltDevice;
    let memory_config = MemoryConfig {
        max_untrusted_advice_size: preprocessing.memory_layout.max_untrusted_advice_size,
        max_trusted_advice_size: preprocessing.memory_layout.max_trusted_advice_size,
        max_input_size: preprocessing.memory_layout.max_input_size,
        max_output_size: preprocessing.memory_layout.max_output_size,
        stack_size: preprocessing.memory_layout.stack_size,
        memory_size: preprocessing.memory_layout.memory_size,
        program_size: Some(preprocessing.memory_layout.program_size),
    };
    let mut io_device = JoltDevice::new(&memory_config);

    io_device.inputs = inputs_bytes.to_vec();
    io_device.outputs = outputs_bytes.to_vec();

    JoltRV64IMAC::verify(
        preprocessing,
        proof,
        io_device,
        trusted_advice_commitment,
        None,
    )
}
