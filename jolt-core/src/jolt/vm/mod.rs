#![allow(clippy::type_complexity)]
#![allow(dead_code)]

use crate::field::JoltField;
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::r1cs::constraints::R1CSConstraints;
use crate::r1cs::spartan::UniformSpartanProof;
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
use tracer::instruction::{RV32IMCycle, RV32IMInstruction};
use tracer::JoltDevice;

use crate::msm::icicle;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::utils::transcript::Transcript;

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltVerifierPreprocessing<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub generators: PCS::Setup,
    pub bytecode: BytecodePreprocessing,
    pub ram: RAMPreprocessing,
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
    // pub opening_proof: ReducedOpeningProof<F, PCS, ProofTranscript>,
}

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
        let ram_preprocessing = RAMPreprocessing::preprocess(memory_init);

        // TODO(moodlezoup): Update for Twist+Shout
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
            ram: ram_preprocessing,
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
        JoltProof<WORD_SIZE, F, PCS, ProofTranscript>,
        // JoltCommitments<PCS, ProofTranscript>,
        JoltDevice,
        Option<ProverDebugInfo<F, ProofTranscript>>,
    ) {
        icicle::icicle_init();
        let trace_length = trace.len();
        println!("Trace length: {trace_length}");

        F::initialize_lookup_tables(std::mem::take(&mut preprocessing.field));

        // TODO(moodlezoup): Truncate generators

        // TODO(JP): Drop padding on number of steps
        let padded_trace_length = trace_length.next_power_of_two();
        let padding = padded_trace_length - trace_length;
        let last_address = trace.last().unwrap().instruction().normalize().address;
        if padding != 0 {
            // Pad with NoOps (with sequential addresses) followed by a final JALR
            trace.extend((0..padding - 1).map(|i| RV32IMCycle::NoOp(last_address + 4 * i)));
            // Final JALR sets NextUnexpandedPC = 0
            trace.push(RV32IMCycle::last_jalr(last_address + 4 * (padding - 1)));
        } else {
            // Replace last JAL with JALR to set NextUnexpandedPC = 0
            assert!(matches!(trace.last().unwrap(), RV32IMCycle::JAL(_)));
            *trace.last_mut().unwrap() = RV32IMCycle::last_jalr(last_address);
        }

        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        let mut opening_accumulator: ProverOpeningAccumulator<F, ProofTranscript> =
            ProverOpeningAccumulator::new();

        Self::fiat_shamir_preamble(
            &mut transcript,
            &program_io,
            &program_io.memory_layout,
            trace_length,
        );

        // jolt_commitments
        //     .read_write_values()
        //     .iter()
        //     .for_each(|value| value.append_to_transcript(&mut transcript));
        // jolt_commitments
        //     .init_final_values()
        //     .iter()
        //     .for_each(|value| value.append_to_transcript(&mut transcript));

        let constraint_builder = Self::Constraints::construct_constraints(padded_trace_length);
        let spartan_key = UniformSpartanProof::<F, ProofTranscript>::setup(
            &constraint_builder,
            padded_trace_length,
        );
        transcript.append_scalar(&spartan_key.vk_digest);

        let mut transcript_1 = transcript.clone();

        let r1cs_proof = UniformSpartanProof::prove::<PCS>(
            &preprocessing,
            &constraint_builder,
            &spartan_key,
            &trace,
            &mut opening_accumulator,
            &mut transcript_1,
        )
        .ok()
        .unwrap();

        let shard_len = std::cmp::min(
            padded_trace_length,
            std::cmp::max(
                1 << (padded_trace_length.log_2() - padded_trace_length.log_2() / 2),
                1 << 20,
            ),
        );

        println!("shard_len: {shard_len}");

        let r1cs_proof = UniformSpartanProof::prove_streaming::<PCS>(
            &preprocessing,
            &constraint_builder,
            &spartan_key,
            &trace,
            shard_len,
            &mut opening_accumulator,
            &mut transcript,
        )
        .ok()
        .unwrap();

        // let shard_len = padded_trace_length;

        // let outer_sumcheck_proof = UniformSpartanProof::prove_streaming::<PCS>(
        //     &preprocessing,
        //     &constraint_builder,
        //     &spartan_key,
        //     &trace,
        //     shard_len,
        //     &mut opening_accumulator,
        //     &mut transcript_1,
        // );
        // .ok()
        // .unwrap();

        // #[cfg(test)]
        // {
        //     println!(
        //         "Streaming sum-check proof has {} polys.",
        //         outer_sumcheck_proof.compressed_polys.len()
        //     );
        //
        //     assert_eq!(
        //         outer_sumcheck_proof.compressed_polys.len(),
        //         r1cs_proof.outer_sumcheck_proof.compressed_polys.len()
        //     );
        //     for i in 0..r1cs_proof.outer_sumcheck_proof.compressed_polys.len() {
        //         assert_eq!(
        //             outer_sumcheck_proof.compressed_polys[i].coeffs_except_linear_term,
        //             r1cs_proof.outer_sumcheck_proof.compressed_polys[i].coeffs_except_linear_term,
        //             "The streaming sum-check errs in round {i}"
        //         );
        //     }
        //     println!("streaming sum-check prover returns the correct proof.");
        // }

        let instruction_proof = LookupsProof::prove(
            &preprocessing.shared.generators,
            &trace,
            &mut opening_accumulator,
            &mut transcript,
        );

        let registers_proof =
            RegistersTwistProof::prove(&trace, &mut opening_accumulator, &mut transcript);

        let ram_proof = RAMTwistProof::prove(
            &preprocessing.shared.ram,
            &trace,
            &program_io,
            1 << 16, // TODO(moodlezoup)
            &mut opening_accumulator,
            &mut transcript,
        );

        let bytecode_proof =
            BytecodeShoutProof::prove(&preprocessing.shared.bytecode, &trace, &mut transcript);

        // Batch-prove all openings
        // let opening_proof =
        //     opening_accumulator.reduce_and_prove::<PCS>(&preprocessing.generators, &mut transcript);

        let jolt_proof = JoltProof {
            trace_length,
            bytecode: bytecode_proof,
            instruction_lookups: instruction_proof,
            ram: ram_proof,
            registers: registers_proof,
            r1cs: r1cs_proof,
            // opening_proof,
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
        preprocessing: JoltVerifierPreprocessing<F, PCS, ProofTranscript>,
        proof: JoltProof<WORD_SIZE, F, PCS, ProofTranscript>,
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

        // Regenerate the uniform Spartan key
        let padded_trace_length = proof.trace_length.next_power_of_two();
        let r1cs_builder = Self::Constraints::construct_constraints(padded_trace_length);
        let spartan_key =
            UniformSpartanProof::<F, ProofTranscript>::setup(&r1cs_builder, padded_trace_length);
        transcript.append_scalar(&spartan_key.vk_digest);

        // commitments
        //     .read_write_values()
        //     .iter()
        //     .for_each(|value| value.append_to_transcript(&mut transcript));
        // commitments
        //     .init_final_values()
        //     .iter()
        //     .for_each(|value| value.append_to_transcript(&mut transcript));

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
            &preprocessing.ram,
            &program_io,
            &mut transcript,
        )?;
        proof.bytecode.verify(
            &preprocessing.bytecode,
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
