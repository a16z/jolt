#![allow(clippy::type_complexity)]

use crate::field::JoltField;
use crate::r1cs::builder::CombinedUniformBuilder;
use crate::r1cs::jolt_constraints::{construct_jolt_constraints, JoltIn};
use crate::r1cs::spartan::{self, UniformSpartanProof};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::log2;
use common::constants::RAM_START_ADDRESS;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use strum::EnumCount;

use crate::jolt::vm::timestamp_range_check::RangeCheckPolynomials;
use crate::jolt::{
    instruction::JoltInstruction, subtable::JoltSubtableSet,
    vm::timestamp_range_check::TimestampValidityProof,
};
use crate::lasso::memory_checking::{MemoryCheckingProver, MemoryCheckingVerifier};
use crate::poly::commitment::commitment_scheme::{BatchType, CommitmentScheme};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::structured_poly::StructuredCommitment;
use crate::r1cs::inputs::{R1CSCommitment, R1CSInputs, R1CSProof};
use crate::utils::errors::ProofVerifyError;
use crate::utils::thread::{drop_in_background_thread, unsafe_allocate_zero_vec};
use crate::utils::transcript::{AppendToTranscript, ProofTranscript};
use common::{
    constants::MEMORY_OPS_PER_INSTRUCTION,
    rv_trace::{ELFInstruction, JoltDevice, MemoryOp},
};

use self::bytecode::BytecodePreprocessing;
use self::instruction_lookups::{
    InstructionCommitment, InstructionLookupsPreprocessing, InstructionLookupsProof,
};
use self::read_write_memory::{
    MemoryCommitment, ReadWriteMemory, ReadWriteMemoryPreprocessing, ReadWriteMemoryProof,
};
use self::timestamp_range_check::RangeCheckCommitment;
use self::{
    bytecode::{BytecodeCommitment, BytecodePolynomials, BytecodeProof, BytecodeRow},
    instruction_lookups::InstructionPolynomials,
};

use super::instruction::JoltInstructionSet;

#[derive(Clone)]
pub struct JoltPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::Setup,
    pub instruction_lookups: InstructionLookupsPreprocessing<F>,
    pub bytecode: BytecodePreprocessing<F>,
    pub read_write_memory: ReadWriteMemoryPreprocessing,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct JoltTraceStep<InstructionSet: JoltInstructionSet> {
    pub instruction_lookup: Option<InstructionSet>,
    pub bytecode_row: BytecodeRow,
    pub memory_ops: [MemoryOp; MEMORY_OPS_PER_INSTRUCTION],
}

impl<InstructionSet: JoltInstructionSet> JoltTraceStep<InstructionSet> {
    fn no_op() -> Self {
        JoltTraceStep {
            instruction_lookup: None,
            bytecode_row: BytecodeRow::no_op(0),
            memory_ops: [
                MemoryOp::noop_read(),  // rs1
                MemoryOp::noop_read(),  // rs2
                MemoryOp::noop_write(), // rd is write-only
                MemoryOp::noop_read(),  // RAM byte 1
                MemoryOp::noop_read(),  // RAM byte 2
                MemoryOp::noop_read(),  // RAM byte 3
                MemoryOp::noop_read(),  // RAM byte 4
            ],
        }
    }

    fn pad(trace: &mut Vec<Self>) {
        let unpadded_length = trace.len();
        let padded_length = unpadded_length.next_power_of_two();
        trace.resize(padded_length, Self::no_op());
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltProof<const C: usize, const M: usize, F, PCS, InstructionSet, Subtables>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    InstructionSet: JoltInstructionSet,
    Subtables: JoltSubtableSet<F>,
{
    pub trace_length: usize,
    pub program_io: JoltDevice,
    pub bytecode: BytecodeProof<F, PCS>,
    pub read_write_memory: ReadWriteMemoryProof<F, PCS>,
    pub instruction_lookups: InstructionLookupsProof<C, M, F, PCS, InstructionSet, Subtables>,
    pub r1cs: R1CSProof<F, PCS>,
}

pub struct JoltPolynomials<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub bytecode: BytecodePolynomials<F, PCS>,
    pub read_write_memory: ReadWriteMemory<F, PCS>,
    pub timestamp_range_check: RangeCheckPolynomials<F, PCS>,
    pub instruction_lookups: InstructionPolynomials<F, PCS>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltCommitments<PCS: CommitmentScheme> {
    pub bytecode: BytecodeCommitment<PCS>,
    pub read_write_memory: MemoryCommitment<PCS>,
    pub timestamp_range_check: RangeCheckCommitment<PCS>,
    pub instruction_lookups: InstructionCommitment<PCS>,
    pub r1cs: Option<R1CSCommitment<PCS>>,
}

impl<PCS: CommitmentScheme> JoltCommitments<PCS> {
    fn append_to_transcript(&self, transcript: &mut ProofTranscript) {
        self.bytecode.append_to_transcript(b"bytecode", transcript);
        self.read_write_memory
            .append_to_transcript(b"read_write_memory", transcript);
        self.timestamp_range_check
            .append_to_transcript(b"timestamp_range_check", transcript);
        self.instruction_lookups
            .append_to_transcript(b"instruction_lookups", transcript);
        self.r1cs
            .as_ref()
            .unwrap()
            .append_to_transcript(b"r1cs", transcript);
    }
}

impl<F, PCS> StructuredCommitment<PCS> for JoltPolynomials<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    type Commitment = JoltCommitments<PCS>;

    #[tracing::instrument(skip_all, name = "JoltPolynomials::commit")]
    fn commit(&self, generators: &PCS::Setup) -> Self::Commitment {
        let bytecode_trace_polys = vec![
            &self.bytecode.a_read_write,
            &self.bytecode.t_read,
            &self.bytecode.v_read_write[0],
            &self.bytecode.v_read_write[1],
            &self.bytecode.v_read_write[2],
            &self.bytecode.v_read_write[3],
            &self.bytecode.v_read_write[4],
            &self.bytecode.v_read_write[5],
        ];
        let num_bytecode_trace_polys = bytecode_trace_polys.len();

        let memory_trace_polys: Vec<&DensePolynomial<F>> = [&self.read_write_memory.a_ram]
            .into_iter()
            .chain(self.read_write_memory.v_read.iter())
            .chain([&self.read_write_memory.v_write_rd].into_iter())
            .chain(self.read_write_memory.v_write_ram.iter())
            .chain(self.read_write_memory.t_read.iter())
            .chain(self.read_write_memory.t_write_ram.iter())
            .collect();
        let num_memory_trace_polys = memory_trace_polys.len();

        let range_check_polys: Vec<&DensePolynomial<F>> = self
            .timestamp_range_check
            .read_cts_read_timestamp
            .iter()
            .chain(self.timestamp_range_check.read_cts_global_minus_read.iter())
            .chain(self.timestamp_range_check.final_cts_read_timestamp.iter())
            .chain(
                self.timestamp_range_check
                    .final_cts_global_minus_read
                    .iter(),
            )
            .collect();
        let num_range_check_polys = range_check_polys.len();

        let instruction_trace_polys: Vec<&DensePolynomial<F>> = self
            .instruction_lookups
            .dim
            .iter()
            .chain(self.instruction_lookups.read_cts.iter())
            .chain(self.instruction_lookups.E_polys.iter())
            .chain(self.instruction_lookups.instruction_flag_polys.iter())
            .chain([&self.instruction_lookups.lookup_outputs].into_iter())
            .collect();

        let all_trace_polys = bytecode_trace_polys
            .into_iter()
            .chain(memory_trace_polys.into_iter())
            .chain(range_check_polys.into_iter())
            .chain(instruction_trace_polys.into_iter())
            .collect::<Vec<_>>();
        let mut trace_comitments =
            PCS::batch_commit_polys_ref(&all_trace_polys, generators, BatchType::Big);

        let bytecode_trace_commitment = trace_comitments
            .drain(..num_bytecode_trace_polys)
            .collect::<Vec<_>>();
        let memory_trace_commitment = trace_comitments
            .drain(..num_memory_trace_polys)
            .collect::<Vec<_>>();
        let range_check_commitment = trace_comitments
            .drain(..num_range_check_polys)
            .collect::<Vec<_>>();
        let instruction_trace_commitment = trace_comitments;

        let bytecode_t_final_commitment = PCS::commit(&self.bytecode.t_final, generators);
        let (memory_v_final_commitment, memory_t_final_commitment) = rayon::join(
            || PCS::commit(&self.read_write_memory.v_final, generators),
            || PCS::commit(&self.read_write_memory.t_final, generators),
        );
        let instruction_final_commitment = PCS::batch_commit_polys(
            &self.instruction_lookups.final_cts,
            generators,
            BatchType::Big,
        );

        JoltCommitments {
            bytecode: BytecodeCommitment {
                trace_commitments: bytecode_trace_commitment,
                t_final_commitment: bytecode_t_final_commitment,
            },
            read_write_memory: MemoryCommitment {
                trace_commitments: memory_trace_commitment,
                v_final_commitment: memory_v_final_commitment,
                t_final_commitment: memory_t_final_commitment,
            },
            timestamp_range_check: RangeCheckCommitment {
                commitments: range_check_commitment,
            },
            instruction_lookups: InstructionCommitment {
                trace_commitment: instruction_trace_commitment,
                final_commitment: instruction_final_commitment,
            },
            r1cs: None,
        }
    }
}

pub trait Jolt<F: JoltField, PCS: CommitmentScheme<Field = F>, const C: usize, const M: usize> {
    type InstructionSet: JoltInstructionSet;
    type Subtables: JoltSubtableSet<F>;

    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    fn preprocess(
        bytecode: Vec<ELFInstruction>,
        memory_init: Vec<(u64, u8)>,
        max_bytecode_size: usize,
        max_memory_address: usize,
        max_trace_length: usize,
    ) -> JoltPreprocessing<F, PCS> {
        let bytecode_commitment_shapes =
            BytecodePolynomials::<F, PCS>::commit_shapes(max_bytecode_size, max_trace_length);
        let ram_commitment_shapes =
            ReadWriteMemory::<F, PCS>::commitment_shapes(max_memory_address, max_trace_length);
        let timestamp_range_check_commitment_shapes =
            TimestampValidityProof::<F, PCS>::commitment_shapes(max_trace_length);

        let instruction_lookups_preprocessing = InstructionLookupsPreprocessing::preprocess::<
            C,
            M,
            Self::InstructionSet,
            Self::Subtables,
        >();
        let instruction_lookups_commitment_shapes = InstructionLookupsProof::<
            C,
            M,
            F,
            PCS,
            Self::InstructionSet,
            Self::Subtables,
        >::commitment_shapes(
            &instruction_lookups_preprocessing,
            max_trace_length,
        );

        let read_write_memory_preprocessing = ReadWriteMemoryPreprocessing::preprocess(memory_init);

        let bytecode_rows: Vec<BytecodeRow> = bytecode
            .iter()
            .map(BytecodeRow::from_instruction::<Self::InstructionSet>)
            .collect();
        let bytecode_preprocessing = BytecodePreprocessing::<F>::preprocess(bytecode_rows);

        let commitment_shapes = [
            bytecode_commitment_shapes,
            ram_commitment_shapes,
            timestamp_range_check_commitment_shapes,
            instruction_lookups_commitment_shapes,
        ]
        .concat();
        let generators = PCS::setup(&commitment_shapes);

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
        mut trace: Vec<JoltTraceStep<Self::InstructionSet>>,
        circuit_flags: Vec<F>,
        preprocessing: JoltPreprocessing<F, PCS>,
    ) -> (
        JoltProof<C, M, F, PCS, Self::InstructionSet, Self::Subtables>,
        JoltCommitments<PCS>,
    ) {
        let trace_length = trace.len();
        let padded_trace_length = trace_length.next_power_of_two();
        println!("Trace length: {}", trace_length);

        JoltTraceStep::pad(&mut trace);

        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        Self::fiat_shamir_preamble(&mut transcript, &program_io, trace_length);

        let instruction_polynomials = InstructionLookupsProof::<
            C,
            M,
            F,
            PCS,
            Self::InstructionSet,
            Self::Subtables,
        >::polynomialize(
            &preprocessing.instruction_lookups, &trace
        );

        let load_store_flags = &instruction_polynomials.instruction_flag_polys[5..10];
        let (memory_polynomials, read_timestamps) = ReadWriteMemory::new(
            &program_io,
            load_store_flags,
            &preprocessing.read_write_memory,
            &trace,
        );

        let (bytecode_polynomials, range_check_polys) = rayon::join(
            || BytecodePolynomials::<F, PCS>::new(&preprocessing.bytecode, &mut trace),
            || RangeCheckPolynomials::<F, PCS>::new(read_timestamps),
        );

        let jolt_polynomials = JoltPolynomials {
            bytecode: bytecode_polynomials,
            read_write_memory: memory_polynomials,
            timestamp_range_check: range_check_polys,
            instruction_lookups: instruction_polynomials,
        };

        let (witness_segments, r1cs_commitments, r1cs_builder) = Self::r1cs_setup(
            padded_trace_length,
            RAM_START_ADDRESS - program_io.memory_layout.ram_witness_offset,
            &trace,
            &jolt_polynomials,
            circuit_flags,
            &preprocessing.generators,
        );

        let mut jolt_commitments = jolt_polynomials.commit(&preprocessing.generators);

        let spartan_key = spartan::UniformSpartanProof::<F, PCS>::setup_precommitted(
            &r1cs_builder,
            padded_trace_length,
        );

        transcript.append_scalar(b"spartan key", &spartan_key.vk_digest);

        jolt_commitments.r1cs = Some(r1cs_commitments);
        jolt_commitments.append_to_transcript(&mut transcript);

        let bytecode_proof = BytecodeProof::prove_memory_checking(
            &preprocessing.generators,
            &preprocessing.bytecode,
            &jolt_polynomials.bytecode,
            &mut transcript,
        );

        let instruction_proof = InstructionLookupsProof::prove(
            &preprocessing.generators,
            &jolt_polynomials.instruction_lookups,
            &preprocessing.instruction_lookups,
            &mut transcript,
        );

        let memory_proof = ReadWriteMemoryProof::prove(
            &preprocessing.generators,
            &preprocessing.read_write_memory,
            &jolt_polynomials,
            &program_io,
            &mut transcript,
        );

        drop_in_background_thread(jolt_polynomials);

        let spartan_proof = UniformSpartanProof::<F, PCS>::prove_precommitted(
            &preprocessing.generators,
            r1cs_builder,
            &spartan_key,
            witness_segments,
            &mut transcript,
        )
        .expect("r1cs proof failed");
        let r1cs_proof = R1CSProof {
            key: spartan_key,
            proof: spartan_proof,
        };

        let jolt_proof = JoltProof {
            trace_length,
            program_io,
            bytecode: bytecode_proof,
            read_write_memory: memory_proof,
            instruction_lookups: instruction_proof,
            r1cs: r1cs_proof,
        };

        (jolt_proof, jolt_commitments)
    }

    #[tracing::instrument(skip_all)]
    fn verify(
        mut preprocessing: JoltPreprocessing<F, PCS>,
        proof: JoltProof<C, M, F, PCS, Self::InstructionSet, Self::Subtables>,
        commitments: JoltCommitments<PCS>,
    ) -> Result<(), ProofVerifyError> {
        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        Self::fiat_shamir_preamble(&mut transcript, &proof.program_io, proof.trace_length);

        // append the digest of vk (which includes R1CS matrices) and the RelaxedR1CSInstance to the transcript
        transcript.append_scalar(b"spartan key", &proof.r1cs.key.vk_digest);

        commitments.append_to_transcript(&mut transcript);

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
            &commitments,
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

    #[tracing::instrument(skip_all)]
    fn verify_instruction_lookups(
        preprocessing: &InstructionLookupsPreprocessing<F>,
        generators: &PCS::Setup,
        proof: InstructionLookupsProof<C, M, F, PCS, Self::InstructionSet, Self::Subtables>,
        commitment: &InstructionCommitment<PCS>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        InstructionLookupsProof::verify(preprocessing, generators, proof, commitment, transcript)
    }

    #[tracing::instrument(skip_all)]
    fn verify_bytecode(
        preprocessing: &BytecodePreprocessing<F>,
        generators: &PCS::Setup,
        proof: BytecodeProof<F, PCS>,
        commitment: &BytecodeCommitment<PCS>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        BytecodeProof::verify_memory_checking(
            preprocessing,
            generators,
            proof,
            commitment,
            transcript,
        )
    }

    #[tracing::instrument(skip_all)]
    fn verify_memory(
        preprocessing: &mut ReadWriteMemoryPreprocessing,
        generators: &PCS::Setup,
        proof: ReadWriteMemoryProof<F, PCS>,
        commitment: &JoltCommitments<PCS>,
        program_io: JoltDevice,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        assert!(program_io.inputs.len() <= program_io.memory_layout.max_input_size as usize);
        assert!(program_io.outputs.len() <= program_io.memory_layout.max_output_size as usize);
        preprocessing.program_io = Some(program_io);

        ReadWriteMemoryProof::verify(proof, generators, preprocessing, commitment, transcript)
    }

    #[tracing::instrument(skip_all)]
    fn verify_r1cs(
        generators: &PCS::Setup,
        proof: R1CSProof<F, PCS>,
        commitments: JoltCommitments<PCS>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        proof
            .verify(generators, commitments, C, transcript)
            .map_err(|e| ProofVerifyError::SpartanError(e.to_string()))
    }

    #[tracing::instrument(skip_all, name = "Jolt::r1cs_setup")]
    fn r1cs_setup(
        padded_trace_length: usize,
        memory_start: u64,
        instructions: &[JoltTraceStep<Self::InstructionSet>],
        polynomials: &JoltPolynomials<F, PCS>,
        circuit_flags: Vec<F>,
        generators: &PCS::Setup,
    ) -> (
        Vec<Vec<F>>,
        R1CSCommitment<PCS>,
        CombinedUniformBuilder<F, JoltIn>,
    ) {
        let inputs = Self::r1cs_construct_inputs(
            padded_trace_length,
            instructions,
            polynomials,
            circuit_flags,
        );
        let mut inputs_flat: Vec<Vec<F>> = inputs.clone_to_trace_len_chunks();

        let builder = construct_jolt_constraints(padded_trace_length, memory_start);
        let aux = builder.compute_aux(&inputs_flat);

        assert_eq!(inputs.chunks_x.len(), inputs.chunks_y.len());
        let span = tracing::span!(tracing::Level::INFO, "commit_chunks_flags");
        let _guard = span.enter();
        let chunk_batch_slices: Vec<&[F]> = [&inputs.chunks_x, &inputs.chunks_y]
            .iter()
            .flat_map(|batchee| batchee.chunks(padded_trace_length))
            .collect();
        let chunks_comms = PCS::batch_commit(&chunk_batch_slices, generators, BatchType::Big);

        let circuit_flags_comms = PCS::batch_commit(
            &inputs
                .circuit_flags_bits
                .chunks(padded_trace_length)
                .collect::<Vec<&[F]>>(),
            generators,
            BatchType::Big,
        );
        drop(_guard);

        let io_comms = PCS::batch_commit(&[inputs.pc.as_ref()], generators, BatchType::Big);
        let aux_comms = PCS::batch_commit(
            &aux.iter().map(AsRef::as_ref).collect::<Vec<&[F]>>(),
            generators,
            BatchType::Big,
        );

        let r1cs_commitments = R1CSCommitment::<PCS> {
            io: io_comms,
            aux: aux_comms,
            chunks: chunks_comms,
            circuit_flags: circuit_flags_comms,
        };

        #[cfg(test)]
        {
            let (az, bz, cz) = builder.compute_spartan_Az_Bz_Cz(&inputs_flat, &aux);
            builder.assert_valid(&az, &bz, &cz);
        }

        inputs_flat.extend(aux);

        (inputs_flat, r1cs_commitments, builder)
    }

    // Assemble the R1CS inputs from across other Jolt structs.
    #[tracing::instrument(skip_all, name = "Jolt::r1cs_construct_inputs")]
    fn r1cs_construct_inputs<'a>(
        padded_trace_length: usize,
        instructions: &'a [JoltTraceStep<Self::InstructionSet>],
        polynomials: &'a JoltPolynomials<F, PCS>,
        circuit_flags: Vec<F>,
    ) -> R1CSInputs<'a, F> {
        let log_M = log2(M) as usize;

        // Derive chunks_x and chunks_y
        let span = tracing::span!(tracing::Level::INFO, "compute_chunks_operands");
        let _guard = span.enter();

        let num_chunks = padded_trace_length * C;
        let mut chunks_x: Vec<F> = unsafe_allocate_zero_vec(num_chunks);
        let mut chunks_y: Vec<F> = unsafe_allocate_zero_vec(num_chunks);

        for (instruction_index, op) in instructions.iter().enumerate() {
            if let Some(instr) = &op.instruction_lookup {
                let (chunks_x_op, chunks_y_op) = instr.operand_chunks(C, log_M);
                for (chunk_index, (x, y)) in chunks_x_op
                    .into_iter()
                    .zip(chunks_y_op.into_iter())
                    .enumerate()
                {
                    let flat_chunk_index = instruction_index + chunk_index * padded_trace_length;
                    chunks_x[flat_chunk_index] = F::from_u64(x).unwrap();
                    chunks_y[flat_chunk_index] = F::from_u64(y).unwrap();
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

        let span = tracing::span!(tracing::Level::INFO, "flatten_instruction_flags");
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

        let inputs: R1CSInputs<F> = R1CSInputs::new(
            padded_trace_length,
            bytecode_a.clone(),
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

        inputs
    }

    fn fiat_shamir_preamble(
        transcript: &mut ProofTranscript,
        program_io: &JoltDevice,
        trace_length: usize,
    ) {
        transcript.append_u64(b"Unpadded trace length", trace_length as u64);
        transcript.append_u64(b"C", C as u64);
        transcript.append_u64(b"M", M as u64);
        transcript.append_u64(b"# instructions", Self::InstructionSet::COUNT as u64);
        transcript.append_u64(b"# subtables", Self::Subtables::COUNT as u64);
        transcript.append_u64(b"Max input size", program_io.memory_layout.max_input_size);
        transcript.append_u64(b"Max output size", program_io.memory_layout.max_output_size);
        transcript.append_bytes(b"Program inputs", &program_io.inputs);
        transcript.append_bytes(b"Program outputs", &program_io.outputs);
        transcript.append_u64(b"Program panic", program_io.panic as u64);
    }
}

pub mod bytecode;
pub mod instruction_lookups;
pub mod read_write_memory;
pub mod rv32i_vm;
pub mod timestamp_range_check;
