#![allow(clippy::type_complexity)]
#![allow(dead_code)]

use crate::field::JoltField;
use crate::jolt::vm::rv32im_vm::Serializable;
#[cfg(feature = "prover")]
use crate::msm::icicle;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::VerifierOpeningAccumulator;
use crate::r1cs::constraints::R1CSConstraints;
use crate::r1cs::spartan::UniformSpartanProof;
use crate::utils::errors::ProofVerifyError;
use crate::utils::transcript::Transcript;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use bytecode::{BytecodePreprocessing, BytecodeShoutProof};
use common::jolt_device::MemoryLayout;
use instruction_lookups::LookupsProof;
use ram::{RAMPreprocessing, RAMTwistProof};
use registers::RegistersTwistProof;
use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};
use tracer::instruction::RV32IMInstruction;
use tracer::JoltDevice;

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltSharedPreprocessing {
    pub bytecode: BytecodePreprocessing,
    pub ram: RAMPreprocessing,
    pub memory_layout: MemoryLayout,
}

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltVerifierPreprocessing<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub generators: PCS::VerifierSetup,
    pub shared: JoltSharedPreprocessing,
}

impl<F, PCS, ProofTranscript> Serializable for JoltVerifierPreprocessing<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
}

impl<F, PCS, ProofTranscript> JoltVerifierPreprocessing<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub fn save_to_target_dir(&self, target_dir: &str) -> std::io::Result<()> {
        let filename = Path::new(target_dir).join("jolt_verifier_preprocessing.dat");
        let mut file = File::create(filename.as_path())?;
        let mut data = Vec::new();
        self.serialize_compressed(&mut data).unwrap();
        file.write_all(&data)?;
        Ok(())
    }

    pub fn read_from_target_dir(target_dir: &str) -> std::io::Result<Self> {
        let filename = Path::new(target_dir).join("jolt_verifier_preprocessing.dat");
        let mut file = File::open(filename.as_path())?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Ok(Self::deserialize_compressed(&*data).unwrap())
    }
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltProverPreprocessing<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub generators: PCS::ProverSetup,
    pub shared: JoltSharedPreprocessing,
    field: F::SmallValueLookupTables,
}

impl<F, PCS, ProofTranscript> From<&JoltProverPreprocessing<F, PCS, ProofTranscript>>
    for JoltVerifierPreprocessing<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    fn from(preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>) -> Self {
        let generators = PCS::setup_verifier(&preprocessing.generators);
        JoltVerifierPreprocessing {
            generators,
            shared: preprocessing.shared.clone(),
        }
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct JoltProof<const WORD_SIZE: usize, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub trace_length: usize,
    pub bytecode: BytecodeShoutProof<F, ProofTranscript>,
    pub instruction_lookups: LookupsProof<WORD_SIZE, F, PCS, ProofTranscript>,
    pub ram: RAMTwistProof<F, ProofTranscript>,
    pub registers: RegistersTwistProof<F, ProofTranscript>,
    pub r1cs: UniformSpartanProof<F, ProofTranscript>,
    pub opening_proof: ReducedOpeningProof<F, PCS, ProofTranscript>,
    pub commitments: JoltCommitments<F, PCS, ProofTranscript>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct JoltCommitments<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub commitments: Vec<PCS::Commitment>,
}

pub trait JoltCommon<const WORD_SIZE: usize, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    fn shared_preprocess(
        bytecode: Vec<RV32IMInstruction>,
        memory_layout: MemoryLayout,
        memory_init: Vec<(u64, u8)>,
    ) -> JoltSharedPreprocessing {
        #[cfg(feature = "prover")]
        icicle::icicle_init();

        // let read_write_memory_preprocessing = ReadWriteMemoryPreprocessing::preprocess(memory_init);
        let bytecode_preprocessing = BytecodePreprocessing::preprocess(bytecode);
        let ram_preprocessing = RAMPreprocessing::preprocess(memory_init);

        JoltSharedPreprocessing {
            memory_layout,
            bytecode: bytecode_preprocessing,
            ram: ram_preprocessing,
        }
    }

    fn fiat_shamir_preamble(
        transcript: &mut ProofTranscript,
        program_io: &JoltDevice,
        memory_layout: &MemoryLayout,
        trace_length: usize,
    ) {
        transcript.append_u64(trace_length as u64);
        transcript.append_u64(WORD_SIZE as u64);
        // transcript.append_u64(Self::InstructionSet::COUNT as u64);
        transcript.append_u64(memory_layout.max_input_size);
        transcript.append_u64(memory_layout.max_output_size);
        transcript.append_bytes(&program_io.inputs);
        transcript.append_bytes(&program_io.outputs);
        transcript.append_u64(program_io.panic as u64);
    }
}

pub trait JoltVerifier<const WORD_SIZE: usize, F, PCS, ProofTranscript>:
    JoltCommon<WORD_SIZE, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    type Constraints: R1CSConstraints<F>;

    #[tracing::instrument(skip_all)]
    fn verify(
        preprocessing: JoltVerifierPreprocessing<F, PCS, ProofTranscript>,
        proof: JoltProof<WORD_SIZE, F, PCS, ProofTranscript>,
        // commitments: JoltCommitments<PCS, ProofTranscript>,
        program_io: JoltDevice,
        #[cfg(feature = "prover")]
        _debug_info: Option<ProverDebugInfo<F, ProofTranscript, PCS>>,
    ) -> Result<(), ProofVerifyError> {
        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        let mut opening_accumulator: VerifierOpeningAccumulator<F, PCS, ProofTranscript> =
            VerifierOpeningAccumulator::new();

        #[cfg(all(test, feature = "prover"))]
        if let Some(debug_info) = _debug_info {
            transcript.compare_to(debug_info.transcript);
            opening_accumulator
                .compare_to(debug_info.opening_accumulator, &debug_info.prover_setup);
        }

        Self::fiat_shamir_preamble(
            &mut transcript,
            &program_io,
            &preprocessing.shared.memory_layout,
            proof.trace_length,
        );

        // Regenerate the uniform Spartan key
        let padded_trace_length = proof.trace_length.next_power_of_two();
        let r1cs_builder = Self::Constraints::construct_constraints(padded_trace_length);
        let spartan_key =
            UniformSpartanProof::<F, ProofTranscript>::setup(&r1cs_builder, padded_trace_length);
        transcript.append_scalar(&spartan_key.vk_digest);

        proof
            .r1cs
            .verify(&spartan_key, &mut opening_accumulator, &mut transcript)
            .map_err(|e| ProofVerifyError::SpartanError(e.to_string()))?;
        proof
            .instruction_lookups
            .verify(&mut opening_accumulator, &mut transcript)?;
        proof
            .registers
            .verify(padded_trace_length, &mut transcript)?;
        proof.ram.verify(
            1 << 16,
            padded_trace_length,
            &preprocessing.shared.ram,
            &program_io,
            &mut transcript,
        )?;
        proof.bytecode.verify(
            &preprocessing.shared.bytecode,
            padded_trace_length,
            &mut transcript,
        )?;

        // Batch-verify all openings
        // opening_accumulator.reduce_and_verify(
        //     &preprocessing.generators,
        //     &proof.opening_proof,
        //     &mut transcript,
        // )?;

        Ok(())
    }
}

pub mod bytecode;
pub mod instruction_lookups;
pub mod output_check;
pub mod ram;
pub mod ram_read_write_checking;
pub mod registers;
pub mod registers_read_write_checking;
pub mod rv32im_vm;
#[cfg(feature = "prover")]
mod prover;
#[cfg(feature = "prover")]
pub use prover::*;

