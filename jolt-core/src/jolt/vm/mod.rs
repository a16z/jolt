#![allow(clippy::type_complexity)]
#![allow(dead_code)]

use crate::field::JoltField;
use crate::jolt::vm::rv32im_vm::Serializable;
#[cfg(feature = "prover")]
use crate::msm::icicle;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
#[cfg(all(test, feature = "prover"))]
use crate::poly::commitment::dory::DoryGlobals;
use crate::poly::opening_proof::{ReducedOpeningProof, VerifierOpeningAccumulator};
use crate::r1cs::constraints::R1CSConstraints;
use crate::r1cs::spartan::UniformSpartanProof;
use crate::utils::errors::ProofVerifyError;
use crate::utils::transcript::Transcript;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use bytecode::{BytecodePreprocessing, BytecodeShoutProof};
use common::jolt_device::{JoltDevice, MemoryLayout};
use instruction_lookups::LookupsProof;
use ram::{RAMPreprocessing, RAMTwistProof};
use registers::RegistersTwistProof;
use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};
use tracer::instruction::RV32IMInstruction;

impl Serializable for JoltDevice {}

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
        ram_K: usize,
    ) {
        transcript.append_u64(trace_length as u64);
        transcript.append_u64(ram_K as u64);
        transcript.append_u64(WORD_SIZE as u64);
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
        mut program_io: JoltDevice,
        #[cfg(feature = "prover")] _debug_info: Option<ProverDebugInfo<F, ProofTranscript, PCS>>,
    ) -> Result<(), ProofVerifyError> {
        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        let mut opening_accumulator: VerifierOpeningAccumulator<F, PCS, ProofTranscript> =
            VerifierOpeningAccumulator::new();

        // truncate trailing zeros on device outputs
        program_io.outputs.truncate(
            program_io
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        #[cfg(all(test, feature = "prover"))]
        {
            if let Some(debug_info) = _debug_info {
                transcript.compare_to(debug_info.transcript);
                opening_accumulator
                    .compare_to(debug_info.opening_accumulator, &debug_info.prover_setup);
            }
        }

        #[cfg(all(test, feature = "prover"))]
        let K = [
            preprocessing.shared.bytecode.code_size,
            proof.ram.K,
            1 << 16, // K for instruction lookups Shout
        ]
        .into_iter()
        .max()
        .unwrap();
        #[cfg(all(test, feature = "prover"))]
        let T = proof.trace_length.next_power_of_two();
        // Need to initialize globals because the verifier computes commitments
        // in `VerifierOpeningAccumulator::append` inside of a `#[cfg(test)]` block
        #[cfg(all(test, feature = "prover"))]
        let _guard = DoryGlobals::initialize(K, T);

        Self::fiat_shamir_preamble(
            &mut transcript,
            &program_io,
            &preprocessing.shared.memory_layout,
            proof.trace_length,
            proof.ram.K,
        );

        for commitment in proof.commitments.commitments.iter() {
            transcript.append_serializable(commitment);
        }

        // Regenerate the uniform Spartan key
        let padded_trace_length = proof.trace_length.next_power_of_two();
        let r1cs_builder = Self::Constraints::construct_constraints(padded_trace_length);
        let spartan_key =
            UniformSpartanProof::<F, ProofTranscript>::setup(&r1cs_builder, padded_trace_length);
        transcript.append_scalar(&spartan_key.vk_digest);

        proof
            .r1cs
            .verify(
                &spartan_key,
                &proof.commitments,
                &mut opening_accumulator,
                &mut transcript,
            )
            .map_err(|e| ProofVerifyError::SpartanError(e.to_string()))?;

        proof.instruction_lookups.verify(
            &proof.commitments,
            &mut opening_accumulator,
            &mut transcript,
        )?;

        proof.registers.verify(
            &proof.commitments,
            padded_trace_length,
            &mut opening_accumulator,
            &mut transcript,
        )?;

        proof.ram.verify(
            padded_trace_length,
            &preprocessing.shared.ram,
            &proof.commitments,
            &program_io,
            &mut transcript,
            &mut opening_accumulator,
        )?;

        proof.bytecode.verify(
            &preprocessing.shared.bytecode,
            &proof.commitments,
            padded_trace_length,
            &mut transcript,
            &mut opening_accumulator,
        )?;

        // Batch-verify all openings
        opening_accumulator.reduce_and_verify(
            &preprocessing.generators,
            &proof.opening_proof,
            &mut transcript,
        )?;

        Ok(())
    }
}

pub mod bytecode;
pub mod instruction_lookups;
pub mod output_check;
#[cfg(feature = "prover")]
mod prover;
pub mod ram;
pub mod registers;
pub mod registers_read_write_checking;
pub mod rv32im_vm;
#[cfg(feature = "prover")]
pub use prover::*;
