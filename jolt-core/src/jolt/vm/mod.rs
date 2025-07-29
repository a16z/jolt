#![allow(clippy::type_complexity)]
#![allow(dead_code)]

use crate::field::JoltField;
use crate::jolt::vm::bytecode::BytecodePreprocessing;
use crate::jolt::vm::ram::NUM_RA_I_VARS;
// use crate::jolt::vm::ram::remap_address;
use crate::jolt::vm::rv32im_vm::Serializable;
use crate::jolt::witness::AllCommittedPolynomials;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::dory::DoryGlobals;
use crate::poly::opening_proof::{
    ProverOpeningAccumulator, ReducedOpeningProof, VerifierOpeningAccumulator,
};
use crate::r1cs::constraints::R1CSConstraints;
use crate::r1cs::spartan::UniformSpartanProof;
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::utils::transcript::Transcript;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::jolt_device::MemoryLayout;
use ram::RAMPreprocessing;
use rayon::prelude::*;
use std::collections::HashMap;
use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};
use tracer::emulator::memory::Memory;
use tracer::instruction::{RV32IMCycle, RV32IMInstruction};
use tracer::JoltDevice;

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltSharedPreprocessing {
    pub bytecode: BytecodePreprocessing,
    pub ram: RAMPreprocessing,
    pub memory_layout: MemoryLayout,
}

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::VerifierSetup,
    pub shared: JoltSharedPreprocessing,
}

impl<F, PCS> Serializable for JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
}

impl<F, PCS> JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
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
pub struct JoltProverPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::ProverSetup,
    pub shared: JoltSharedPreprocessing,
    field: F::SmallValueLookupTables,
}

impl<F, PCS> Serializable for JoltProverPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
}

impl<F, PCS> JoltProverPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
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

impl<F, PCS> From<&JoltProverPreprocessing<F, PCS>> for JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn from(preprocessing: &JoltProverPreprocessing<F, PCS>) -> Self {
        let generators = PCS::setup_verifier(&preprocessing.generators);
        JoltVerifierPreprocessing {
            generators,
            shared: preprocessing.shared.clone(),
        }
    }
}

pub struct ProverDebugInfo<F, ProofTranscript, PCS>
where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    pub(crate) transcript: ProofTranscript,
    pub(crate) opening_accumulator: ProverOpeningAccumulator<F>,
    pub(crate) prover_setup: PCS::ProverSetup,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct JoltProof<const WORD_SIZE: usize, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
{
    pub trace_length: usize,
    // pub instruction_lookups: LookupsProof<WORD_SIZE, F, PCS, ProofTranscript>,
    // pub ram: RAMTwistProof<F, ProofTranscript>,
    // pub registers: RegistersTwistProof<F, ProofTranscript>,
    pub r1cs: UniformSpartanProof<F, ProofTranscript>,
    pub opening_proof: ReducedOpeningProof<F, PCS, ProofTranscript>,
    pub commitments: JoltCommitments<F, PCS>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct JoltCommitments<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub commitments: Vec<PCS::Commitment>,
}

pub trait Jolt<const WORD_SIZE: usize, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    ProofTranscript: Transcript,
{
    type Constraints: R1CSConstraints<F>;

    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    fn shared_preprocess(
        bytecode: Vec<RV32IMInstruction>,
        memory_layout: MemoryLayout,
        memory_init: Vec<(u64, u8)>,
    ) -> JoltSharedPreprocessing {
        let bytecode_preprocessing = BytecodePreprocessing::preprocess(bytecode);
        let ram_preprocessing = RAMPreprocessing::preprocess(memory_init);

        JoltSharedPreprocessing {
            memory_layout,
            bytecode: bytecode_preprocessing,
            ram: ram_preprocessing,
        }
    }

    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    fn prover_preprocess(
        bytecode: Vec<RV32IMInstruction>,
        memory_layout: MemoryLayout,
        memory_init: Vec<(u64, u8)>,
        _max_bytecode_size: usize,
        _max_memory_size: usize,
        max_trace_length: usize,
    ) -> JoltProverPreprocessing<F, PCS> {
        let small_value_lookup_tables = F::compute_lookup_tables();
        F::initialize_lookup_tables(small_value_lookup_tables.clone());

        let shared = Self::shared_preprocess(bytecode, memory_layout, memory_init);

        let max_K = 1 << NUM_RA_I_VARS;
        let max_T = max_trace_length.next_power_of_two();

        println!("setup...");
        // TODO(moodlezoup): Change setup parameter to # variables everywhere
        let generators = PCS::setup_prover(max_K.log_2() + max_T.log_2());
        println!("setup done");

        JoltProverPreprocessing {
            generators,
            shared,
            field: small_value_lookup_tables,
        }
    }

    #[tracing::instrument(skip_all, name = "Jolt::prove")]
    fn prove(
        mut program_io: JoltDevice,
        mut trace: Vec<RV32IMCycle>,
        _final_memory_state: Memory,
        mut preprocessing: JoltProverPreprocessing<F, PCS>,
    ) -> (
        JoltProof<WORD_SIZE, F, PCS, ProofTranscript>,
        JoltDevice,
        Option<ProverDebugInfo<F, ProofTranscript, PCS>>,
    ) {
        let trace_length = trace.len();
        println!("Trace length: {trace_length}");

        // truncate trailing zeros on device outputs
        program_io.outputs.truncate(
            program_io
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        F::initialize_lookup_tables(std::mem::take(&mut preprocessing.field));

        // TODO(moodlezoup): Truncate generators

        // TODO(JP): Drop padding on number of steps
        let padded_trace_length = (trace_length + 1).next_power_of_two();
        trace.resize(padded_trace_length, RV32IMCycle::NoOp);

        let ram_addresses: Vec<usize> = trace
            .par_iter()
            .map(|_cycle| {
                todo!()
                // remap_address(
                //     cycle.ram_access().address() as u64,
                //     &preprocessing.shared.memory_layout,
                // ) as usize
            })
            .collect();
        let ram_K = ram_addresses.par_iter().max().unwrap().next_power_of_two();

        let K = [
            preprocessing.shared.bytecode.code_size,
            ram_K,
            1 << 16, // K for instruction lookups Shout
        ]
        .into_iter()
        .max()
        .unwrap();
        println!("T = {padded_trace_length}, K = {K}");

        let _guard = DoryGlobals::initialize(K, padded_trace_length);

        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        let mut opening_accumulator: ProverOpeningAccumulator<F> = ProverOpeningAccumulator::new();

        Self::fiat_shamir_preamble(
            &mut transcript,
            &program_io,
            &program_io.memory_layout,
            trace_length,
            1 << 16, // TODO(moodlezoup)
        );

        let committed_polys: Vec<_> = AllCommittedPolynomials::par_iter()
            .map(|poly| poly.generate_witness(&preprocessing, &trace))
            .collect();
        let commitments: Vec<_> = committed_polys
            .par_iter()
            .map(|poly| PCS::commit(poly, &preprocessing.generators).0)
            .collect();
        for commitment in commitments.iter() {
            transcript.append_serializable(commitment);
        }

        let constraint_builder = Self::Constraints::construct_constraints(padded_trace_length);
        let spartan_key = UniformSpartanProof::<F, ProofTranscript>::setup(
            &constraint_builder,
            padded_trace_length,
        );
        transcript.append_scalar(&spartan_key.vk_digest);

        let r1cs_proof = UniformSpartanProof::prove::<PCS>(
            &preprocessing,
            &constraint_builder,
            spartan_key,
            &trace,
            &mut opening_accumulator,
            &mut transcript,
        )
        .ok()
        .unwrap();

        // let instruction_proof = LookupsProof::prove(
        //     &preprocessing,
        //     &trace,
        //     &mut opening_accumulator,
        //     &mut transcript,
        // );

        // let registers_proof = RegistersTwistProof::prove(
        //     &preprocessing,
        //     &trace,
        //     &mut opening_accumulator,
        //     &mut transcript,
        // );

        // let ram_proof = RAMTwistProof::prove(
        //     &preprocessing,
        //     &trace,
        //     final_memory_state,
        //     &program_io,
        //     ram_K,
        //     &mut opening_accumulator,
        //     &mut transcript,
        // );

        // let bytecode_proof = BytecodeShoutProof::prove(
        //     &preprocessing,
        //     &trace,
        //     &mut opening_accumulator,
        //     &mut transcript,
        // );

        // Batch-prove all openings
        let mut polynomials_map = HashMap::new();
        for polynomial in AllCommittedPolynomials::iter() {
            polynomials_map.insert(
                *polynomial,
                polynomial.generate_witness(&preprocessing, &trace),
            );
        }
        let opening_proof = opening_accumulator.reduce_and_prove(
            polynomials_map,
            HashMap::new(),
            &preprocessing.generators,
            &mut transcript,
        );

        let jolt_proof = JoltProof {
            trace_length,
            // instruction_lookups: instruction_proof,
            // ram: ram_proof,
            // registers: registers_proof,
            r1cs: r1cs_proof,
            opening_proof,
            commitments: JoltCommitments { commitments },
        };

        #[cfg(test)]
        let debug_info = Some(ProverDebugInfo {
            transcript,
            opening_accumulator,
            prover_setup: preprocessing.generators.clone(),
        });
        #[cfg(not(test))]
        let debug_info = None;

        (jolt_proof, program_io, debug_info)
    }

    #[tracing::instrument(skip_all)]
    fn verify(
        preprocessing: JoltVerifierPreprocessing<F, PCS>,
        proof: JoltProof<WORD_SIZE, F, PCS, ProofTranscript>,
        mut program_io: JoltDevice,
        _debug_info: Option<ProverDebugInfo<F, ProofTranscript, PCS>>,
    ) -> Result<(), ProofVerifyError> {
        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        let mut opening_accumulator: VerifierOpeningAccumulator<F> =
            VerifierOpeningAccumulator::new();

        // truncate trailing zeros on device outputs
        program_io.outputs.truncate(
            program_io
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        #[cfg(test)]
        {
            if let Some(debug_info) = _debug_info {
                transcript.compare_to(debug_info.transcript);
                opening_accumulator.compare_to(debug_info.opening_accumulator);
            }
        }

        #[cfg(test)]
        let K = [
            preprocessing.shared.bytecode.code_size,
            // proof.ram.K,
            1 << 16, // K for instruction lookups Shout
        ]
        .into_iter()
        .max()
        .unwrap();
        #[cfg(test)]
        let T = proof.trace_length.next_power_of_two();
        // Need to initialize globals because the verifier computes commitments
        // in `VerifierOpeningAccumulator::append` inside of a `#[cfg(test)]` block
        #[cfg(test)]
        let _guard = DoryGlobals::initialize(K, T);

        Self::fiat_shamir_preamble(
            &mut transcript,
            &program_io,
            &preprocessing.shared.memory_layout,
            proof.trace_length,
            // proof.ram.K,
            1 << 16, // TODO(moodlezoup)
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
                spartan_key,
                &proof.commitments,
                &mut opening_accumulator,
                &mut transcript,
            )
            .map_err(|e| ProofVerifyError::SpartanError(e.to_string()))?;

        // proof.instruction_lookups.verify(
        //     &proof.commitments,
        //     &mut opening_accumulator,
        //     &mut transcript,
        // )?;

        // proof.registers.verify(
        //     &proof.commitments,
        //     padded_trace_length,
        //     &mut opening_accumulator,
        //     &mut transcript,
        // )?;

        // proof.ram.verify(
        //     padded_trace_length,
        //     &preprocessing.shared.ram,
        //     &proof.commitments,
        //     &program_io,
        //     &mut transcript,
        //     &mut opening_accumulator,
        // )?;

        // proof.bytecode.verify(
        //     &preprocessing.shared.bytecode,
        //     &proof.commitments,
        //     padded_trace_length,
        //     &mut transcript,
        //     &mut opening_accumulator,
        // )?;

        // Batch-verify all openings
        let mut commitments_map = HashMap::new();
        for polynomial in AllCommittedPolynomials::iter() {
            commitments_map.insert(
                *polynomial,
                proof.commitments.commitments[polynomial.to_index()].clone(),
            );
        }
        opening_accumulator.reduce_and_verify(
            &preprocessing.generators,
            &mut commitments_map,
            &proof.opening_proof,
            &mut transcript,
        )?;

        Ok(())
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

pub mod bytecode;
pub mod instruction_lookups;
pub mod ram;
pub mod registers;
pub mod rv32im_vm;
