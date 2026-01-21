use super::program::Program;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::commitment_scheme::StreamingCommitmentScheme;
use crate::poly::commitment::dory::DoryCommitmentScheme;
use crate::transcripts::Transcript;
use crate::zkvm::proof_serialization::JoltProof;
use crate::zkvm::prover::JoltProverPreprocessing;
use crate::zkvm::ProverDebugInfo;
use common::jolt_device::MemoryLayout;
use tracer::JoltDevice;

#[allow(clippy::type_complexity)]
#[cfg(feature = "prover")]
pub fn preprocess(
    guest: &Program,
    max_trace_length: usize,
) -> JoltProverPreprocessing<ark_bn254::Fr, DoryCommitmentScheme> {
    use crate::zkvm::verifier::JoltSharedPreprocessing;

    let (bytecode, memory_init, program_size) = guest.decode();

    let mut memory_config = guest.memory_config;
    memory_config.program_size = Some(program_size);
    let memory_layout = MemoryLayout::new(&memory_config);
    let shared_preprocessing =
        JoltSharedPreprocessing::new(bytecode, memory_layout, memory_init, max_trace_length);
    JoltProverPreprocessing::new(shared_preprocessing)
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
#[cfg(feature = "prover")]
pub fn prove<F: JoltField, PCS: StreamingCommitmentScheme<Field = F>, FS: Transcript>(
    guest: &Program,
    inputs_bytes: &[u8],
    untrusted_advice_bytes: &[u8],
    trusted_advice_bytes: &[u8],
    trusted_advice_commitment: Option<<PCS as CommitmentScheme>::Commitment>,
    trusted_advice_hint: Option<<PCS as CommitmentScheme>::OpeningProofHint>,
    output_bytes: &mut [u8],
    preprocessing: &JoltProverPreprocessing<F, PCS>,
) -> (
    JoltProof<F, PCS, FS>,
    JoltDevice,
    Option<ProverDebugInfo<F, FS, PCS>>,
) {
    use crate::zkvm::prover::JoltCpuProver;

    let prover = JoltCpuProver::gen_from_elf(
        preprocessing,
        &guest.elf_contents,
        inputs_bytes,
        untrusted_advice_bytes,
        trusted_advice_bytes,
        trusted_advice_commitment,
        trusted_advice_hint,
    );
    let io_device = prover.program_io.clone();
    let (proof, debug_info) = prover.prove();
    output_bytes[..io_device.outputs.len()].copy_from_slice(&io_device.outputs);
    (proof, io_device, debug_info)
}
