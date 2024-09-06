#![allow(clippy::type_complexity)]

use crate::field::JoltField;
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::r1cs::constraints::{JoltRV32IMConstraints, R1CSConstraints};
use crate::r1cs::spartan::{self, UniformSpartanProof};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::RAM_START_ADDRESS;
use common::rv_trace::NUM_CIRCUIT_FLAGS;
use serde::{Deserialize, Serialize};
use strum::EnumCount;
use timestamp_range_check::TimestampRangeCheckStuff;

use crate::jolt::vm::timestamp_range_check::TimestampRangeCheckPolynomials;
use crate::jolt::{
    instruction::{
        div::DIVInstruction, divu::DIVUInstruction, mulh::MULHInstruction,
        mulhsu::MULHSUInstruction, rem::REMInstruction, remu::REMUInstruction,
        VirtualInstructionSequence,
    },
    subtable::JoltSubtableSet,
    vm::timestamp_range_check::TimestampValidityProof,
};
use crate::lasso::memory_checking::{
    MemoryCheckingProver, MemoryCheckingVerifier, NoAdditionalWitness,
};
use crate::poly::commitment::commitment_scheme::{BatchType, CommitmentScheme};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::r1cs::inputs::{ConstraintInput, R1CSPolynomials, R1CSProof};
use crate::utils::errors::ProofVerifyError;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::{AppendToTranscript, ProofTranscript};
use common::{
    constants::MEMORY_OPS_PER_INSTRUCTION,
    rv_trace::{ELFInstruction, JoltDevice, MemoryOp},
};

use self::bytecode::{
    BytecodeCommitments, BytecodePolynomials, BytecodePreprocessing, BytecodeProof, BytecodeRow,
    BytecodeStuff,
};
use self::instruction_lookups::{
    InstructionLookupCommitments, InstructionLookupPolynomials, InstructionLookupStuff,
    InstructionLookupsPreprocessing, InstructionLookupsProof,
};
use self::read_write_memory::{
    ReadWriteMemoryCommitments, ReadWriteMemoryPolynomials, ReadWriteMemoryPreprocessing,
    ReadWriteMemoryProof, ReadWriteMemoryStuff,
};
use self::timestamp_range_check::TimestampRangeCheckCommitments;

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
    pub circuit_flags: [bool; NUM_CIRCUIT_FLAGS],
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
            circuit_flags: [false; NUM_CIRCUIT_FLAGS],
        }
    }

    fn pad(trace: &mut Vec<Self>) {
        let unpadded_length = trace.len();
        let padded_length = unpadded_length.next_power_of_two();
        trace.resize(padded_length, Self::no_op());
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltProof<const C: usize, const M: usize, I, F, PCS, InstructionSet, Subtables>
where
    I: ConstraintInput,
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
    pub r1cs: R1CSProof<C, I, F>,
}

struct JoltStuff<T: Sync> {
    bytecode: BytecodeStuff<T>,
    read_write_memory: ReadWriteMemoryStuff<T>,
    instruction_lookups: InstructionLookupStuff<T>,
    timestamp_range_check: TimestampRangeCheckStuff<T>,
}

pub struct JoltPolynomials<F: JoltField> {
    pub bytecode: BytecodePolynomials<F>,
    pub read_write_memory: ReadWriteMemoryPolynomials<F>,
    pub timestamp_range_check: TimestampRangeCheckPolynomials<F>,
    pub instruction_lookups: InstructionLookupPolynomials<F>,
    pub r1cs: R1CSPolynomials<F>,
}

impl<F: JoltField> JoltPolynomials<F> {
    pub fn r1cs_witness_value<const C: usize, I: ConstraintInput>(&self, index: usize) -> F {
        let trace_len = self.bytecode.v_read_write[0].len();
        I::from_index::<C>(index / trace_len).get_poly_ref(self)[index % trace_len]
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltCommitments<PCS: CommitmentScheme> {
    pub bytecode: BytecodeCommitments<PCS>,
    pub read_write_memory: ReadWriteMemoryCommitments<PCS>,
    pub timestamp_range_check: TimestampRangeCheckCommitments<PCS>,
    pub instruction_lookups: InstructionLookupCommitments<PCS>,
    pub r1cs: Vec<PCS::Commitment>,
}

impl<PCS: CommitmentScheme> JoltCommitments<PCS> {
    fn append_to_transcript(&self, transcript: &mut ProofTranscript) {
        self.bytecode.append_to_transcript(transcript);
        self.read_write_memory.append_to_transcript(transcript);
        self.timestamp_range_check.append_to_transcript(transcript);
        self.instruction_lookups.append_to_transcript(transcript);
        for commitment in &self.r1cs {
            commitment.append_to_transcript(transcript);
        }
    }
}

impl<F, PCS> StructuredCommitment<PCS> for JoltPolynomials<F>
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
        let num_instruction_polys = instruction_trace_polys.len();

        let aux_polys = self.r1cs.aux.as_ref().unwrap();
        let r1cs_trace_polys: Vec<&DensePolynomial<F>> = self
            .r1cs
            .chunks_x
            .iter()
            .chain(self.r1cs.chunks_y.iter())
            .chain(self.r1cs.circuit_flags.iter())
            // .chain(todo!("aux polys"))
            .collect();

        let all_trace_polys = bytecode_trace_polys
            .into_iter()
            .chain(memory_trace_polys.into_iter())
            .chain(range_check_polys.into_iter())
            .chain(instruction_trace_polys.into_iter())
            .chain(r1cs_trace_polys.into_iter())
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
        let instruction_trace_commitment = trace_comitments
            .drain(..num_instruction_polys)
            .collect::<Vec<_>>();
        let r1cs_trace_commitment = trace_comitments;

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
            bytecode: BytecodeCommitments {
                trace_commitments: bytecode_trace_commitment,
                t_final_commitment: bytecode_t_final_commitment,
            },
            read_write_memory: ReadWriteMemoryCommitments {
                trace_commitments: memory_trace_commitment,
                v_final_commitment: memory_v_final_commitment,
                t_final_commitment: memory_t_final_commitment,
            },
            timestamp_range_check: TimestampRangeCheckCommitments {
                commitments: range_check_commitment,
            },
            instruction_lookups: InstructionLookupCommitments {
                trace_commitment: instruction_trace_commitment,
                final_commitment: instruction_final_commitment,
            },
            r1cs: r1cs_trace_commitment,
        }
    }
}

pub trait Jolt<F: JoltField, PCS: CommitmentScheme<Field = F>, const C: usize, const M: usize> {
    type InstructionSet: JoltInstructionSet;
    type Subtables: JoltSubtableSet<F>;
    type Constraints: R1CSConstraints<C, F>;

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
        let ram_commitment_shapes = ReadWriteMemoryPolynomials::<F, PCS>::commitment_shapes(
            max_memory_address,
            max_trace_length,
        );
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
            .into_iter()
            .flat_map(|instruction| match instruction.opcode {
                tracer::RV32IM::MULH => MULHInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::MULHSU => MULHSUInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::DIV => DIVInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::DIVU => DIVUInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::REM => REMInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::REMU => REMUInstruction::<32>::virtual_sequence(instruction),
                _ => vec![instruction],
            })
            .map(|instruction| BytecodeRow::from_instruction::<Self::InstructionSet>(&instruction))
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
        preprocessing: JoltPreprocessing<F, PCS>,
    ) -> (
        JoltProof<
            C,
            M,
            <Self::Constraints as R1CSConstraints<C, F>>::Inputs,
            F,
            PCS,
            Self::InstructionSet,
            Self::Subtables,
        >,
        JoltCommitments<PCS>,
    ) {
        let trace_length = trace.len();
        let padded_trace_length = trace_length.next_power_of_two();
        println!("Trace length: {}", trace_length);

        JoltTraceStep::pad(&mut trace);

        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        Self::fiat_shamir_preamble(&mut transcript, &program_io, trace_length);

        let (instruction_polynomials, instruction_flag_bitvectors) = InstructionLookupsProof::<
            C,
            M,
            F,
            PCS,
            Self::InstructionSet,
            Self::Subtables,
        >::generate_witness(
            &preprocessing.instruction_lookups,
            &trace,
        );

        let load_store_flags = &instruction_polynomials.instruction_flag_polys[5..10];
        let (memory_polynomials, read_timestamps) = ReadWriteMemoryPolynomials::new(
            &program_io,
            load_store_flags,
            &preprocessing.read_write_memory,
            &trace,
        );

        let (bytecode_polynomials, range_check_polys) = rayon::join(
            || BytecodePolynomials::<F, PCS>::new(&preprocessing.bytecode, &mut trace),
            || TimestampRangeCheckPolynomials::<F, PCS>::new(read_timestamps),
        );

        let r1cs_builder = Self::Constraints::construct_constraints(
            padded_trace_length,
            RAM_START_ADDRESS - program_io.memory_layout.ram_witness_offset,
        );
        let spartan_key = spartan::UniformSpartanProof::<
            C,
            <Self::Constraints as R1CSConstraints<C, F>>::Inputs,
            F,
        >::setup_precommitted(&r1cs_builder, padded_trace_length);

        let r1cs_polynomials = R1CSPolynomials::new::<
            C,
            M,
            Self::InstructionSet,
            <Self::Constraints as R1CSConstraints<C, F>>::Inputs,
        >(&trace);

        let mut jolt_polynomials = JoltPolynomials {
            bytecode: bytecode_polynomials,
            read_write_memory: memory_polynomials,
            timestamp_range_check: range_check_polys,
            instruction_lookups: instruction_polynomials,
            r1cs: r1cs_polynomials,
        };

        r1cs_builder.compute_aux(&mut jolt_polynomials);

        let jolt_commitments = jolt_polynomials.commit(&preprocessing.generators);

        transcript.append_scalar(&spartan_key.vk_digest);

        jolt_commitments.append_to_transcript(&mut transcript);

        let mut opening_accumulator: ProverOpeningAccumulator<'_, F> =
            ProverOpeningAccumulator::new();

        let bytecode_proof = BytecodeProof::prove_memory_checking(
            &preprocessing.generators,
            &preprocessing.bytecode,
            &jolt_polynomials.bytecode,
            &NoAdditionalWitness,
            &mut opening_accumulator,
            &mut transcript,
        );

        let instruction_proof = InstructionLookupsProof::prove(
            &preprocessing.generators,
            &jolt_polynomials.instruction_lookups,
            instruction_flag_bitvectors,
            &preprocessing.instruction_lookups,
            &mut opening_accumulator,
            &mut transcript,
        );

        let memory_proof = ReadWriteMemoryProof::prove(
            &preprocessing.generators,
            &preprocessing.read_write_memory,
            &jolt_polynomials.read_write_memory,
            &jolt_polynomials.bytecode,
            &program_io,
            &mut opening_accumulator,
            &mut transcript,
        );

        let spartan_proof = UniformSpartanProof::<
            C,
            <Self::Constraints as R1CSConstraints<C, F>>::Inputs,
            F,
        >::prove_new(
            &r1cs_builder,
            &spartan_key,
            &jolt_polynomials,
            &mut opening_accumulator,
            &mut transcript,
        )
        .expect("r1cs proof failed");

        let r1cs_proof = R1CSProof {
            key: spartan_key,
            proof: spartan_proof,
        };

        println!("{} openings accumulated", opening_accumulator.len());
        opening_accumulator.reduce_and_prove::<PCS>(&preprocessing.generators, &mut transcript);

        drop_in_background_thread(jolt_polynomials);

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
        proof: JoltProof<
            C,
            M,
            <Self::Constraints as R1CSConstraints<C, F>>::Inputs,
            F,
            PCS,
            Self::InstructionSet,
            Self::Subtables,
        >,
        commitments: JoltCommitments<PCS>,
    ) -> Result<(), ProofVerifyError> {
        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        Self::fiat_shamir_preamble(&mut transcript, &proof.program_io, proof.trace_length);

        // append the digest of vk (which includes R1CS matrices) and the RelaxedR1CSInstance to the transcript
        transcript.append_scalar(&proof.r1cs.key.vk_digest);

        commitments.append_to_transcript(&mut transcript);

        let mut opening_accumulator: VerifierOpeningAccumulator<'_, F, PCS> =
            VerifierOpeningAccumulator::new();

        Self::verify_bytecode(
            &preprocessing.bytecode,
            &preprocessing.generators,
            proof.bytecode,
            &commitments.bytecode,
            &mut opening_accumulator,
            &mut transcript,
        )?;
        Self::verify_instruction_lookups(
            &preprocessing.instruction_lookups,
            &preprocessing.generators,
            proof.instruction_lookups,
            &commitments.instruction_lookups,
            &mut opening_accumulator,
            &mut transcript,
        )?;
        Self::verify_memory(
            &mut preprocessing.read_write_memory,
            &preprocessing.generators,
            proof.read_write_memory,
            &commitments,
            proof.program_io,
            &mut opening_accumulator,
            &mut transcript,
        )?;
        Self::verify_r1cs(proof.r1cs, &mut opening_accumulator, &mut transcript)?;

        todo!("verify batched opening");
        Ok(())
    }

    #[tracing::instrument(skip_all)]
    fn verify_instruction_lookups<'a>(
        preprocessing: &InstructionLookupsPreprocessing<F>,
        generators: &PCS::Setup,
        proof: InstructionLookupsProof<C, M, F, PCS, Self::InstructionSet, Self::Subtables>,
        commitment: &'a InstructionLookupCommitments<PCS>,
        opening_accumulator: &mut VerifierOpeningAccumulator<'a, F, PCS>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        InstructionLookupsProof::verify(
            preprocessing,
            generators,
            proof,
            commitment,
            opening_accumulator,
            transcript,
        )
    }

    #[tracing::instrument(skip_all)]
    fn verify_bytecode<'a>(
        preprocessing: &BytecodePreprocessing<F>,
        generators: &PCS::Setup,
        proof: BytecodeProof<F, PCS>,
        commitment: &'a BytecodeCommitments<PCS>,
        opening_accumulator: &mut VerifierOpeningAccumulator<'a, F, PCS>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        BytecodeProof::verify_memory_checking(
            preprocessing,
            generators,
            proof,
            commitment,
            opening_accumulator,
            transcript,
        )
    }

    #[tracing::instrument(skip_all)]
    fn verify_memory<'a>(
        preprocessing: &mut ReadWriteMemoryPreprocessing,
        generators: &PCS::Setup,
        proof: ReadWriteMemoryProof<F, PCS>,
        commitment: &'a JoltCommitments<PCS>,
        program_io: JoltDevice,
        opening_accumulator: &mut VerifierOpeningAccumulator<'a, F, PCS>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        assert!(program_io.inputs.len() <= program_io.memory_layout.max_input_size as usize);
        assert!(program_io.outputs.len() <= program_io.memory_layout.max_output_size as usize);
        preprocessing.program_io = Some(program_io);

        ReadWriteMemoryProof::verify(
            proof,
            generators,
            preprocessing,
            commitment,
            opening_accumulator,
            transcript,
        )
    }

    #[tracing::instrument(skip_all)]
    fn verify_r1cs<'a>(
        proof: R1CSProof<C, <Self::Constraints as R1CSConstraints<C, F>>::Inputs, F>,
        opening_accumulator: &mut VerifierOpeningAccumulator<'a, F, PCS>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        proof
            .verify(transcript)
            .map_err(|e| ProofVerifyError::SpartanError(e.to_string()))
    }

    fn fiat_shamir_preamble(
        transcript: &mut ProofTranscript,
        program_io: &JoltDevice,
        trace_length: usize,
    ) {
        transcript.append_u64(trace_length as u64);
        transcript.append_u64(C as u64);
        transcript.append_u64(M as u64);
        transcript.append_u64(Self::InstructionSet::COUNT as u64);
        transcript.append_u64(Self::Subtables::COUNT as u64);
        transcript.append_u64(program_io.memory_layout.max_input_size);
        transcript.append_u64(program_io.memory_layout.max_output_size);
        transcript.append_bytes(&program_io.inputs);
        transcript.append_bytes(&program_io.outputs);
        transcript.append_u64(program_io.panic as u64);
    }
}

pub mod bytecode;
pub mod instruction_lookups;
pub mod read_write_memory;
pub mod rv32i_vm;
pub mod timestamp_range_check;
