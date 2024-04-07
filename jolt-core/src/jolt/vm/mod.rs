use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::log2;
use common::constants::NUM_R1CS_POLYS;
use common::rv_trace::JoltDevice;
use itertools::max;
use merlin::Transcript;
use rayon::prelude::*;

use crate::jolt::{
    instruction::JoltInstruction, subtable::JoltSubtableSet,
    vm::timestamp_range_check::TimestampValidityProof,
};
use crate::lasso::memory_checking::{MemoryCheckingProver, MemoryCheckingVerifier};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::hyrax::HyraxCommitment;
use crate::poly::pedersen::PedersenGenerators;
use crate::poly::structured_poly::StructuredCommitment;
use crate::r1cs::snark::{R1CSInputs, R1CSProof, R1CSUniqueCommitments};
use crate::r1cs::spartan::UniformSpartanKey;
use crate::utils::errors::ProofVerifyError;
use crate::utils::thread::{drop_in_background_thread, unsafe_allocate_zero_vec};
use common::{
    constants::{MAX_INPUT_SIZE, MAX_OUTPUT_SIZE, MEMORY_OPS_PER_INSTRUCTION},
    rv_trace::{ELFInstruction, MemoryOp},
};

use self::bytecode::BytecodePreprocessing;
use self::instruction_lookups::{
    InstructionCommitment, InstructionLookupsPreprocessing, InstructionLookupsProof,
};
use self::read_write_memory::{
    MemoryCommitment, ReadWriteMemory, ReadWriteMemoryPreprocessing, ReadWriteMemoryProof,
};
use self::{
    bytecode::{BytecodeCommitment, BytecodePolynomials, BytecodeProof, BytecodeRow},
    instruction_lookups::InstructionPolynomials,
};

use super::instruction::JoltInstructionSet;

#[derive(Clone)]
pub struct JoltPreprocessing<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    pub generators: PedersenGenerators<G>,
    pub instruction_lookups: InstructionLookupsPreprocessing<F>,
    pub bytecode: BytecodePreprocessing<F>,
    pub read_write_memory: ReadWriteMemoryPreprocessing,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltProof<const C: usize, const M: usize, F, G, InstructionSet, Subtables>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
    InstructionSet: JoltInstructionSet,
    Subtables: JoltSubtableSet<F>,
{
    pub program_io: JoltDevice,
    pub bytecode: BytecodeProof<F, G>,
    pub read_write_memory: ReadWriteMemoryProof<F, G>,
    pub instruction_lookups: InstructionLookupsProof<C, M, F, G, InstructionSet, Subtables>,
    pub r1cs: R1CSProof<F, G>,
}

pub struct JoltPolynomials<F, G>
where
    F: PrimeField,
    G: CurveGroup<ScalarField = F>,
{
    pub bytecode: BytecodePolynomials<F, G>,
    pub read_write_memory: ReadWriteMemory<F, G>,
    pub instruction_lookups: InstructionPolynomials<F, G>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltCommitments<G: CurveGroup> {
    pub bytecode: BytecodeCommitment<G>,
    pub read_write_memory: MemoryCommitment<G>,
    pub instruction_lookups: InstructionCommitment<G>,
    pub r1cs: R1CSUniqueCommitments<G>,
}

pub trait Jolt<F: PrimeField, G: CurveGroup<ScalarField = F>, const C: usize, const M: usize> {
    type InstructionSet: JoltInstructionSet;
    type Subtables: JoltSubtableSet<F>;

    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    fn preprocess(
        bytecode: Vec<ELFInstruction>,
        memory_init: Vec<(u64, u8)>,
        max_bytecode_size: usize,
        max_memory_address: usize,
        max_trace_length: usize,
    ) -> JoltPreprocessing<F, G> {
        let num_bytecode_generators =
            BytecodePolynomials::<F, G>::num_generators(max_bytecode_size, max_trace_length);
        let num_read_write_memory_generators =
            ReadWriteMemory::<F, G>::num_generators(max_memory_address, max_trace_length);
        let timestamp_range_check_generators =
            TimestampValidityProof::<F, G>::num_generators(max_trace_length);
        let instruction_lookups_preprocessing = InstructionLookupsPreprocessing::preprocess::<
            C,
            M,
            Self::InstructionSet,
            Self::Subtables,
        >();
        let num_instruction_lookup_generators = InstructionLookupsProof::<
            C,
            M,
            F,
            G,
            Self::InstructionSet,
            Self::Subtables,
        >::num_generators(
            &instruction_lookups_preprocessing,
            max_trace_length,
        );

        let read_write_memory_preprocessing = ReadWriteMemoryPreprocessing::preprocess(memory_init);

        let bytecode_rows: Vec<BytecodeRow> = bytecode
            .iter()
            .map(BytecodeRow::from_instruction::<Self::InstructionSet>)
            .collect();
        let bytecode_preprocessing = BytecodePreprocessing::preprocess(bytecode_rows);

        let max_num_generators = max([
            num_bytecode_generators,
            num_read_write_memory_generators,
            timestamp_range_check_generators,
            num_instruction_lookup_generators,
        ])
        .unwrap();
        let generators = PedersenGenerators::new(max_num_generators, b"Jolt v1 Hyrax generators");

        JoltPreprocessing {
            generators,
            instruction_lookups: instruction_lookups_preprocessing,
            bytecode: bytecode_preprocessing,
            read_write_memory: read_write_memory_preprocessing,
        }
    }

    #[tracing::instrument(skip_all, name = "Jolt::prove")]
    fn prove(
        program_io: JoltDevice,
        bytecode_trace: Vec<BytecodeRow>,
        memory_trace: Vec<[MemoryOp; MEMORY_OPS_PER_INSTRUCTION]>,
        instructions: Vec<Option<Self::InstructionSet>>,
        circuit_flags: Vec<F>,
        preprocessing: JoltPreprocessing<F, G>,
    ) -> (
        JoltProof<C, M, F, G, Self::InstructionSet, Self::Subtables>,
        JoltCommitments<G>,
    ) {
        let trace_length = instructions.len();
        let padded_trace_length = trace_length.next_power_of_two();
        println!("Trace length: {}", trace_length);

        let mut transcript = Transcript::new(b"Jolt transcript");

        let bytecode_polynomials: BytecodePolynomials<F, G> =
            BytecodePolynomials::new(&preprocessing.bytecode, bytecode_trace);

        let instruction_polynomials = InstructionLookupsProof::<
            C,
            M,
            F,
            G,
            Self::InstructionSet,
            Self::Subtables,
        >::polynomialize(
            &preprocessing.instruction_lookups, &instructions
        );

        let mut padded_memory_trace = memory_trace;
        padded_memory_trace.resize(
            padded_trace_length,
            [
                MemoryOp::noop_read(),  // rs1
                MemoryOp::noop_read(),  // rs2
                MemoryOp::noop_write(), // rd is write-only
                MemoryOp::noop_read(),  // RAM byte 1
                MemoryOp::noop_read(),  // RAM byte 2
                MemoryOp::noop_read(),  // RAM byte 3
                MemoryOp::noop_read(),  // RAM byte 4
            ],
        );

        let load_store_flags = &instruction_polynomials.instruction_flag_polys[5..10];
        let (memory_polynomials, read_timestamps) = ReadWriteMemory::new(
            &program_io,
            load_store_flags,
            &preprocessing.read_write_memory,
            padded_memory_trace,
            &mut transcript,
        );

        let jolt_polynomials = JoltPolynomials {
            bytecode: bytecode_polynomials,
            read_write_memory: memory_polynomials,
            instruction_lookups: instruction_polynomials,
        };

        let bytecode_commitment =
            BytecodePolynomials::commit(&jolt_polynomials.bytecode, &preprocessing.generators);
        let instruction_commitment = jolt_polynomials
            .instruction_lookups
            .commit(&preprocessing.generators);
        let memory_commitment = ReadWriteMemory::commit(
            &jolt_polynomials.read_write_memory,
            &preprocessing.generators,
        );

        let (spartan_key, witness_segments, r1cs_commitments) = Self::r1cs_bullshit(
            padded_trace_length,
            &instructions,
            &jolt_polynomials,
            circuit_flags,
            &preprocessing.generators,
        );

        let jolt_commitments = JoltCommitments {
            bytecode: bytecode_commitment,
            read_write_memory: memory_commitment,
            instruction_lookups: instruction_commitment,
            r1cs: r1cs_commitments,
        };

        let bytecode_proof = BytecodeProof::prove_memory_checking(
            &preprocessing.bytecode,
            &jolt_polynomials.bytecode,
            &mut transcript,
        );

        let instruction_proof = InstructionLookupsProof::prove(
            &jolt_polynomials.instruction_lookups,
            &preprocessing.instruction_lookups,
            &mut transcript,
        );

        let memory_proof = ReadWriteMemoryProof::prove(
            &preprocessing.read_write_memory,
            &jolt_polynomials.read_write_memory,
            read_timestamps,
            &program_io,
            &preprocessing.generators,
            &mut transcript,
        );

        drop_in_background_thread(jolt_polynomials);

        let r1cs_proof =
            R1CSProof::prove(spartan_key, witness_segments, &mut transcript).expect("proof failed");

        let jolt_proof = JoltProof {
            program_io,
            bytecode: bytecode_proof,
            read_write_memory: memory_proof,
            instruction_lookups: instruction_proof,
            r1cs: r1cs_proof,
        };

        (jolt_proof, jolt_commitments)
    }

    fn verify(
        mut preprocessing: JoltPreprocessing<F, G>,
        proof: JoltProof<C, M, F, G, Self::InstructionSet, Self::Subtables>,
        commitments: JoltCommitments<G>,
    ) -> Result<(), ProofVerifyError> {
        let mut transcript = Transcript::new(b"Jolt transcript");
        Self::verify_bytecode(
            &preprocessing.bytecode,
            &preprocessing.generators,
            proof.bytecode,
            &commitments.bytecode,
            &mut transcript,
        )?;
        Self::verify_instruction_lookups(
            &preprocessing.instruction_lookups,
            &preprocessing.generators,
            proof.instruction_lookups,
            &commitments.instruction_lookups,
            &mut transcript,
        )?;
        Self::verify_memory(
            &mut preprocessing.read_write_memory,
            &preprocessing.generators,
            proof.read_write_memory,
            &commitments.read_write_memory,
            proof.program_io,
            &mut transcript,
        )?;
        Self::verify_r1cs(
            &preprocessing.generators,
            proof.r1cs,
            commitments,
            &mut transcript,
        )?;
        Ok(())
    }

    fn verify_instruction_lookups(
        preprocessing: &InstructionLookupsPreprocessing<F>,
        generators: &PedersenGenerators<G>,
        proof: InstructionLookupsProof<C, M, F, G, Self::InstructionSet, Self::Subtables>,
        commitment: &InstructionCommitment<G>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        InstructionLookupsProof::verify(preprocessing, generators, proof, commitment, transcript)
    }

    fn verify_bytecode(
        preprocessing: &BytecodePreprocessing<F>,
        generators: &PedersenGenerators<G>,
        proof: BytecodeProof<F, G>,
        commitment: &BytecodeCommitment<G>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        BytecodeProof::verify_memory_checking(
            preprocessing,
            generators,
            proof,
            &commitment,
            transcript,
        )
    }

    fn verify_memory(
        preprocessing: &mut ReadWriteMemoryPreprocessing,
        generators: &PedersenGenerators<G>,
        proof: ReadWriteMemoryProof<F, G>,
        commitment: &MemoryCommitment<G>,
        program_io: JoltDevice,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        assert!(program_io.inputs.len() <= MAX_INPUT_SIZE as usize);
        assert!(program_io.outputs.len() <= MAX_OUTPUT_SIZE as usize);
        preprocessing.program_io = Some(program_io);

        ReadWriteMemoryProof::verify(proof, generators, preprocessing, commitment, transcript)
    }

    fn verify_r1cs(
        generators: &PedersenGenerators<G>,
        proof: R1CSProof<F, G>,
        commitments: JoltCommitments<G>,
        transcript: &mut Transcript,
    ) -> Result<(), ProofVerifyError> {
        proof
            .verify(generators, commitments, C, transcript)
            .map_err(|e| ProofVerifyError::SpartanError(e.to_string()))
    }

    fn r1cs_bullshit(
        padded_trace_length: usize,
        instructions: &Vec<Option<Self::InstructionSet>>,
        polynomials: &JoltPolynomials<F, G>,
        circuit_flags: Vec<F>,
        generators: &PedersenGenerators<G>,
    ) -> (UniformSpartanKey<F>, Vec<Vec<F>>, R1CSUniqueCommitments<G>) {
        let log_M = log2(M) as usize;

        /* Assemble the polynomials and commitments from the rest of Jolt.
        The ones that are extra, just for R1CS are:
            - chunks_x
            - chunks_y
            - lookup_output
            - circuit_flags
        */

        // Derive chunks_x and chunks_y
        let span = tracing::span!(tracing::Level::INFO, "compute_chunks_operands");
        let _guard = span.enter();

        let num_chunks = padded_trace_length * C;
        let mut chunks_x: Vec<F> = unsafe_allocate_zero_vec(num_chunks);
        let mut chunks_y: Vec<F> = unsafe_allocate_zero_vec(num_chunks);

        for (instruction_index, op) in instructions.iter().enumerate() {
            if let Some(op) = op {
                let [chunks_x_op, chunks_y_op] = op.operand_chunks(C, log_M);
                for (chunk_index, (x, y)) in chunks_x_op
                    .into_iter()
                    .zip(chunks_y_op.into_iter())
                    .enumerate()
                {
                    let flat_chunk_index = instruction_index + chunk_index * padded_trace_length;
                    chunks_x[flat_chunk_index] = F::from_u64(x as u64).unwrap();
                    chunks_y[flat_chunk_index] = F::from_u64(y as u64).unwrap();
                }
            } else {
                for chunk_index in 0..C {
                    let flat_chunk_index = instruction_index + chunk_index * padded_trace_length;
                    chunks_x[flat_chunk_index] = F::zero();
                    chunks_y[flat_chunk_index] = F::zero();
                }
            }
        }

        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "flatten instruction_flags");
        let _enter = span.enter();
        let instruction_flags: Vec<F> =
            DensePolynomial::flatten(&polynomials.instruction_lookups.instruction_flag_polys);
        drop(_enter);
        drop(span);

        let (bytecode_a, bytecode_v) = polynomials.bytecode.get_polys_r1cs();
        let (memreg_a_rw, memreg_v_reads, memreg_v_writes) =
            polynomials.read_write_memory.get_polys_r1cs();

        let span = tracing::span!(tracing::Level::INFO, "chunks_query");
        let _guard = span.enter();
        let mut chunks_query: Vec<F> =
            Vec::with_capacity(C * polynomials.instruction_lookups.dim[0].len());
        for i in 0..C {
            chunks_query.par_extend(
                polynomials.instruction_lookups.dim[i]
                    .evals_ref()
                    .par_iter(),
            );
        }
        drop(_guard);

        // Commit to R1CS specific items
        let commit_to_chunks = |data: &Vec<F>| -> Vec<HyraxCommitment<NUM_R1CS_POLYS, G>> {
            data.par_chunks(padded_trace_length)
                .map(|chunk| HyraxCommitment::commit_slice(chunk, generators))
                .collect()
        };

        let span = tracing::span!(tracing::Level::INFO, "new_commitments");
        let _guard = span.enter();
        let chunks_x_comms = commit_to_chunks(&chunks_x);
        let chunks_y_comms = commit_to_chunks(&chunks_y);
        let circuit_flags_comm = commit_to_chunks(&circuit_flags);
        drop(_guard);

        // Flattening this out into a Vec<F> and chunking into padded_trace_length-sized chunks
        // will be the exact witness vector to feed into the R1CS
        // after pre-pending IO and appending the AUX
        let inputs: R1CSInputs<F> = R1CSInputs::new(
            padded_trace_length,
            bytecode_a,
            bytecode_v,
            memreg_a_rw,
            memreg_v_reads,
            memreg_v_writes,
            chunks_x,
            chunks_y,
            chunks_query,
            polynomials.instruction_lookups.lookup_outputs.evals(),
            circuit_flags,
            instruction_flags,
        );

        let (spartan_key, witness_segments, io_aux_commitments) =
            R1CSProof::<F, G>::compute_witness_commit(
                32,
                C,
                padded_trace_length,
                &inputs,
                generators,
            )
            .expect("R1CSProof setup failed");

        let r1cs_commitments = R1CSUniqueCommitments::new(
            io_aux_commitments,
            chunks_x_comms,
            chunks_y_comms,
            circuit_flags_comm,
        );

        (spartan_key, witness_segments, r1cs_commitments)
    }
}

pub mod bytecode;
pub mod instruction_lookups;
pub mod read_write_memory;
pub mod rv32i_vm;
pub mod timestamp_range_check;
