use super::program::Program;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::dory::DoryCommitmentScheme;
use crate::transcripts::Transcript;
use crate::zkvm::dag::proof_serialization::JoltProof;
use crate::zkvm::{Jolt, JoltProverPreprocessing, JoltRV32IM, ProverDebugInfo};
use common::jolt_device::MemoryLayout;
use tracer::JoltDevice;

#[allow(clippy::type_complexity)]
#[cfg(feature = "prover")]
pub fn preprocess(
    guest: &Program,
    max_trace_length: usize,
) -> JoltProverPreprocessing<ark_bn254::Fr, DoryCommitmentScheme> {
    let (bytecode, memory_init, program_size) = guest.decode();

    let mut memory_config = guest.memory_config;
    memory_config.program_size = Some(program_size);
    let memory_layout = MemoryLayout::new(&memory_config);

    JoltRV32IM::prover_preprocess(bytecode, memory_layout, memory_init, max_trace_length)
}

#[allow(clippy::type_complexity)]
#[cfg(feature = "prover")]
pub fn prove<F, PCS, FS>(
    guest: &Program,
    inputs_bytes: &[u8],
    output_bytes: &mut [u8],
    preprocessing: &JoltProverPreprocessing<F, PCS>,
) -> (
    JoltProof<F, PCS, FS>,
    JoltDevice,
    Option<ProverDebugInfo<F, FS, PCS>>,
)
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
    JoltRV32IM: Jolt<F, PCS, FS>,
{
    let (proof, io_device, debug_info) =
        JoltRV32IM::prove(preprocessing, &guest.elf_contents, inputs_bytes);
    output_bytes[..io_device.outputs.len()].copy_from_slice(&io_device.outputs);
    (proof, io_device, debug_info)
}
