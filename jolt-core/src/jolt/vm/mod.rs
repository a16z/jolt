#![allow(clippy::type_complexity)]
#![allow(dead_code)]

use crate::field::JoltField;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::opening_proof::{
    ProverOpeningAccumulator, ReducedOpeningProof, VerifierOpeningAccumulator,
};
use crate::r1cs::constraints::R1CSConstraints;
use crate::r1cs::spartan::UniformSpartanProof;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use bytecode::{BytecodePreprocessing, BytecodeShoutProof};
use common::instruction::NUM_CIRCUIT_FLAGS;
use common::memory::MemoryLayout;
use instruction_lookups::LookupsProof;
use ram::RAMTwistProof;
use registers::RegistersTwistProof;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};
use strum::EnumCount;
use tracer::instruction::{RV32IMCycle, RV32IMInstruction};
use tracer::JoltDevice;

use crate::lasso::memory_checking::{
    Initializable, MemoryCheckingProver, MemoryCheckingVerifier, StructuredPolynomialData,
};
use crate::msm::icicle;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::r1cs::inputs::{ConstraintInput, R1CSPolynomials, R1CSProof, R1CSStuff};
use crate::utils::errors::ProofVerifyError;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::{AppendToTranscript, Transcript};
use common::{constants::MEMORY_OPS_PER_INSTRUCTION, memory::MemoryOp};

use super::lookup_table::LookupTables;

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltVerifierPreprocessing<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub generators: PCS::Setup,
    pub bytecode: BytecodePreprocessing,
    // pub read_write_memory: ReadWriteMemoryPreprocessing,
    pub memory_layout: MemoryLayout,
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
    pub shared: JoltVerifierPreprocessing<F, PCS, ProofTranscript>,
    field: F::SmallValueLookupTables,
}

impl<F, PCS, ProofTranscript> JoltProverPreprocessing<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub fn save_to_target_dir(&self, target_dir: &str) -> std::io::Result<()> {
        let filename = Path::new(target_dir).join("jolt_prover_preprocessing.dat");
        let mut file = File::create(filename.as_path())?;
        let mut data = Vec::new();
        self.serialize_compressed(&mut data).unwrap();
        file.write_all(&data)?;
        Ok(())
    }

    pub fn read_from_target_dir(target_dir: &str) -> std::io::Result<Self> {
        let filename = Path::new(target_dir).join("jolt_prover_preprocessing.dat");
        let mut file = File::open(filename.as_path())?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Ok(Self::deserialize_compressed(&*data).unwrap())
    }
}

pub struct ProverDebugInfo<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    pub(crate) transcript: ProofTranscript,
    pub(crate) opening_accumulator: ProverOpeningAccumulator<F, ProofTranscript>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltProof<const WORD_SIZE: usize, I, F, PCS, ProofTranscript>
where
    I: ConstraintInput,
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub trace_length: usize,
    pub bytecode: BytecodeShoutProof<F, ProofTranscript>,
    // pub read_write_memory: ReadWriteMemoryProof<F, PCS, ProofTranscript>,
    pub instruction_lookups: LookupsProof<WORD_SIZE, F, PCS, ProofTranscript>,
    pub ram: RAMTwistProof<F, ProofTranscript>,
    pub registers: RegistersTwistProof<F, ProofTranscript>,
    // pub r1cs: UniformSpartanProof<C, I, F, ProofTranscript>,
    // pub opening_proof: ReducedOpeningProof<F, PCS, ProofTranscript>,
    _marker: PhantomData<I>,
}

#[derive(Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltStuff<T: CanonicalSerialize + CanonicalDeserialize + Sync> {
    // pub(crate) bytecode: BytecodeStuff<T>,
    // pub(crate) read_write_memory: ReadWriteMemoryStuff<T>,
    // pub(crate) timestamp_range_check: TimestampRangeCheckStuff<T>,
    pub(crate) r1cs: R1CSStuff<T>,
}

/// Note –– F: JoltField bound is not enforced.
///
/// See issue #112792 <https://github.com/rust-lang/rust/issues/112792>.
/// Adding #![feature(lazy_type_alias)] to the crate attributes seem to break
/// `alloy_sol_types`.
pub type JoltPolynomials<F: JoltField> = JoltStuff<MultilinearPolynomial<F>>;
/// Note –– PCS: CommitmentScheme bound is not enforced.
///
/// See issue #112792 <https://github.com/rust-lang/rust/issues/112792>.
/// Adding #![feature(lazy_type_alias)] to the crate attributes seem to break
/// `alloy_sol_types`.
pub type JoltCommitments<PCS: CommitmentScheme<ProofTranscript>, ProofTranscript: Transcript> =
    JoltStuff<PCS::Commitment>;

// impl<F: JoltField> JoltPolynomials<F> {
//     #[tracing::instrument(skip_all, name = "JoltPolynomials::commit")]
//     pub fn commit<const C: usize, PCS, ProofTranscript>(
//         &self,
//         preprocessing: &JoltPreprocessing<C, F, PCS, ProofTranscript>,
//     ) -> JoltCommitments<PCS, ProofTranscript>
//     where
//         PCS: CommitmentScheme<ProofTranscript, Field = F>,
//         ProofTranscript: Transcript,
//     {
//         let span = tracing::span!(tracing::Level::INFO, "commit::initialize");
//         let _guard = span.enter();
//         let mut commitments = JoltCommitments::<PCS, ProofTranscript>::initialize(preprocessing);
//         drop(_guard);
//         drop(span);

//         let trace_polys = self.read_write_values();
//         let trace_commitments = PCS::batch_commit(&trace_polys, &preprocessing.generators);

//         commitments
//             .read_write_values_mut()
//             .into_iter()
//             .zip(trace_commitments.into_iter())
//             .for_each(|(dest, src)| *dest = src);

//         commitments
//     }
// }

pub trait Jolt<const WORD_SIZE: usize, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    // type InstructionSet: JoltInstructionSet;
    type Constraints: R1CSConstraints<F>;

    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    fn verifier_preprocess(
        bytecode: Vec<RV32IMInstruction>,
        memory_layout: MemoryLayout,
        memory_init: Vec<(u64, u8)>,
        max_bytecode_size: usize,
        max_memory_address: usize,
        max_trace_length: usize,
    ) -> JoltVerifierPreprocessing<F, PCS, ProofTranscript> {
        icicle::icicle_init();

        // let read_write_memory_preprocessing = ReadWriteMemoryPreprocessing::preprocess(memory_init);

        let bytecode_preprocessing = BytecodePreprocessing::preprocess(bytecode);

        let max_poly_len: usize = [
            (max_bytecode_size + 1).next_power_of_two(), // Account for no-op prepended to bytecode
            max_trace_length.next_power_of_two(),
            max_memory_address.next_power_of_two(),
        ]
        .into_iter()
        .max()
        .unwrap();
        let generators = PCS::setup(max_poly_len);

        JoltVerifierPreprocessing {
            generators,
            memory_layout,
            bytecode: bytecode_preprocessing,
            // read_write_memory: read_write_memory_preprocessing,
            // instruction_lookups: instruction_lookups_preprocessing,
            // read_write_memory: read_write_memory_preprocessing,
        }
    }

    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    fn prover_preprocess(
        bytecode: Vec<RV32IMInstruction>,
        memory_layout: MemoryLayout,
        memory_init: Vec<(u64, u8)>,
        max_bytecode_size: usize,
        max_memory_address: usize,
        max_trace_length: usize,
    ) -> JoltProverPreprocessing<F, PCS, ProofTranscript> {
        let small_value_lookup_tables = F::compute_lookup_tables();
        F::initialize_lookup_tables(small_value_lookup_tables.clone());

        let shared = Self::verifier_preprocess(
            bytecode,
            memory_layout,
            memory_init,
            max_bytecode_size,
            max_memory_address,
            max_trace_length,
        );

        JoltProverPreprocessing {
            shared,
            field: small_value_lookup_tables,
        }
    }

    #[tracing::instrument(skip_all, name = "Jolt::prove")]
    fn prove(
        program_io: JoltDevice,
        mut trace: Vec<RV32IMCycle>,
        mut preprocessing: JoltProverPreprocessing<F, PCS, ProofTranscript>,
    ) -> (
        JoltProof<
            WORD_SIZE,
            <Self::Constraints as R1CSConstraints<F>>::Inputs,
            F,
            PCS,
            ProofTranscript,
        >,
        // JoltCommitments<PCS, ProofTranscript>,
        JoltDevice,
        Option<ProverDebugInfo<F, ProofTranscript>>,
    ) {
        icicle::icicle_init();
        let trace_length = trace.len();
        println!("Trace length: {}", trace_length);

        F::initialize_lookup_tables(std::mem::take(&mut preprocessing.field));

        // TODO(moodlezoup): Truncate generators

        // TODO(JP): Drop padding on number of steps
        trace.resize(trace_length.next_power_of_two(), RV32IMCycle::NoOp);

        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        Self::fiat_shamir_preamble(
            &mut transcript,
            &program_io,
            &program_io.memory_layout,
            trace_length,
        );

        // transcript.append_scalar(&spartan_key.vk_digest);

        // jolt_commitments
        //     .read_write_values()
        //     .iter()
        //     .for_each(|value| value.append_to_transcript(&mut transcript));
        // jolt_commitments
        //     .init_final_values()
        //     .iter()
        //     .for_each(|value| value.append_to_transcript(&mut transcript));

        let mut opening_accumulator: ProverOpeningAccumulator<F, ProofTranscript> =
            ProverOpeningAccumulator::new();

        let bytecode_proof =
            BytecodeShoutProof::prove(&preprocessing.shared.bytecode, &trace, &mut transcript);

        let instruction_proof = LookupsProof::prove(
            &preprocessing.shared.generators,
            &trace,
            &mut opening_accumulator,
            &mut transcript,
        );

        let ram_proof = RAMTwistProof::prove(
            // &preprocessing.generators,
            &trace,
            &program_io,
            1 << 16, // TODO(moodlezoup)
            &mut opening_accumulator,
            &mut transcript,
        );

        let registers_proof =
            RegistersTwistProof::prove(&trace, &mut opening_accumulator, &mut transcript);

        // Batch-prove all openings
        // let opening_proof =
        //     opening_accumulator.reduce_and_prove::<PCS>(&preprocessing.generators, &mut transcript);

        let jolt_proof = JoltProof {
            trace_length,
            bytecode: bytecode_proof,
            instruction_lookups: instruction_proof,
            ram: ram_proof,
            registers: registers_proof,
            // r1cs: spartan_proof,
            // opening_proof,
            _marker: PhantomData,
        };

        #[cfg(test)]
        let debug_info = Some(ProverDebugInfo {
            transcript,
            opening_accumulator,
        });
        #[cfg(not(test))]
        let debug_info = None;
        (jolt_proof, program_io, debug_info)
    }

    #[tracing::instrument(skip_all)]
    fn verify(
        mut preprocessing: JoltVerifierPreprocessing<F, PCS, ProofTranscript>,
        proof: JoltProof<
            WORD_SIZE,
            <Self::Constraints as R1CSConstraints<F>>::Inputs,
            F,
            PCS,
            ProofTranscript,
        >,
        // commitments: JoltCommitments<PCS, ProofTranscript>,
        program_io: JoltDevice,
        _debug_info: Option<ProverDebugInfo<F, ProofTranscript>>,
    ) -> Result<(), ProofVerifyError> {
        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        let mut opening_accumulator: VerifierOpeningAccumulator<F, PCS, ProofTranscript> =
            VerifierOpeningAccumulator::new();

        #[cfg(test)]
        if let Some(debug_info) = _debug_info {
            transcript.compare_to(debug_info.transcript);
            opening_accumulator
                .compare_to(debug_info.opening_accumulator, &preprocessing.generators);
        }
        Self::fiat_shamir_preamble(
            &mut transcript,
            &program_io,
            &preprocessing.memory_layout,
            proof.trace_length,
        );

        // // Regenerate the uniform Spartan key
        // let padded_trace_length = proof.trace_length.next_power_of_two();
        // let memory_start = preprocessing.memory_layout.input_start;
        // let r1cs_builder =
        //     Self::Constraints::construct_constraints(padded_trace_length, memory_start);
        // let spartan_key = spartan::UniformSpartanProof::<C, _, F, ProofTranscript>::setup(
        //     &r1cs_builder,
        //     padded_trace_length,
        // );
        // transcript.append_scalar(&spartan_key.vk_digest);

        // commitments
        //     .read_write_values()
        //     .iter()
        //     .for_each(|value| value.append_to_transcript(&mut transcript));
        // commitments
        //     .init_final_values()
        //     .iter()
        //     .for_each(|value| value.append_to_transcript(&mut transcript));

        Self::verify_instruction_lookups(
            &preprocessing.generators,
            proof.instruction_lookups,
            // &commitments,
            &mut opening_accumulator,
            &mut transcript,
        )?;
        // Self::verify_memory(
        //     &mut preprocessing.read_write_memory,
        //     &preprocessing.generators,
        //     &preprocessing.memory_layout,
        //     proof.read_write_memory,
        //     &commitments,
        //     proof.program_io,
        //     &mut opening_accumulator,
        //     &mut transcript,
        // )?;

        // Batch-verify all openings
        // opening_accumulator.reduce_and_verify(
        //     &preprocessing.generators,
        //     &proof.opening_proof,
        //     &mut transcript,
        // )?;

        Ok(())
    }

    #[tracing::instrument(skip_all)]
    fn verify_instruction_lookups<'a>(
        generators: &PCS::Setup,
        proof: LookupsProof<WORD_SIZE, F, PCS, ProofTranscript>,
        // commitments: &'a JoltCommitments<PCS, ProofTranscript>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        proof.verify(opening_accumulator, transcript)
    }

    // #[allow(clippy::too_many_arguments)]
    // #[tracing::instrument(skip_all)]
    // fn verify_memory<'a>(
    //     preprocessing: &mut ReadWriteMemoryPreprocessing,
    //     generators: &PCS::Setup,
    //     memory_layout: &MemoryLayout,
    //     proof: ReadWriteMemoryProof<F, PCS, ProofTranscript>,
    //     commitment: &'a JoltCommitments<PCS, ProofTranscript>,
    //     program_io: JoltDevice,
    //     opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
    //     transcript: &mut ProofTranscript,
    // ) -> Result<(), ProofVerifyError> {
    //     assert!(program_io.inputs.len() <= memory_layout.max_input_size as usize);
    //     assert!(program_io.outputs.len() <= memory_layout.max_output_size as usize);
    //     // pair the memory layout with the program io from the proof
    //     preprocessing.program_io = Some(JoltDevice {
    //         inputs: program_io.inputs,
    //         outputs: program_io.outputs,
    //         panic: program_io.panic,
    //         memory_layout: memory_layout.clone(),
    //     });

    //     ReadWriteMemoryProof::verify(
    //         proof,
    //         generators,
    //         preprocessing,
    //         commitment,
    //         opening_accumulator,
    //         transcript,
    //     )
    // }

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

pub mod bytecode;
pub mod instruction_lookups;
pub mod ram;
pub mod registers;
pub mod rv32i_vm;
