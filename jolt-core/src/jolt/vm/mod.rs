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
    MemoryCheckingProver, MemoryCheckingVerifier, NoAdditionalWitness, StructuredPolynomialData,
};
use crate::poly::commitment::commitment_scheme::{BatchType, CommitmentScheme};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::r1cs::inputs::{ConstraintInput, R1CSPolynomials, R1CSProof, R1CSStuff};
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
    r1cs: R1CSStuff<T>,
}

impl<T: Sync> StructuredPolynomialData<T> for JoltStuff<T> {
    fn read_write_values(&self) -> Vec<&T> {
        self.bytecode
            .read_write_values()
            .into_iter()
            .chain(self.read_write_memory.read_write_values())
            .chain(self.instruction_lookups.read_write_values())
            .chain(self.timestamp_range_check.read_write_values())
            .chain(self.r1cs.read_write_values())
            .collect()
    }

    fn init_final_values(&self) -> Vec<&T> {
        self.bytecode
            .init_final_values()
            .into_iter()
            .chain(self.read_write_memory.init_final_values())
            .chain(self.instruction_lookups.init_final_values())
            .chain(self.timestamp_range_check.init_final_values())
            .chain(self.r1cs.init_final_values())
            .collect()
    }

    fn read_write_values_mut(&mut self) -> Vec<&mut T> {
        self.bytecode
            .read_write_values_mut()
            .into_iter()
            .chain(self.read_write_memory.read_write_values_mut())
            .chain(self.instruction_lookups.read_write_values_mut())
            .chain(self.timestamp_range_check.read_write_values_mut())
            .chain(self.r1cs.read_write_values_mut())
            .collect()
    }

    fn init_final_values_mut(&mut self) -> Vec<&mut T> {
        self.bytecode
            .init_final_values_mut()
            .into_iter()
            .chain(self.read_write_memory.init_final_values_mut())
            .chain(self.instruction_lookups.init_final_values_mut())
            .chain(self.timestamp_range_check.init_final_values_mut())
            .chain(self.r1cs.init_final_values_mut())
            .collect()
    }
}

pub type JoltPolynomials<F: JoltField> = JoltStuff<DensePolynomial<F>>;
pub type JoltCommitments<PCS: CommitmentScheme> = JoltStuff<PCS::Commitment>;

impl<F: JoltField> JoltPolynomials<F> {
    pub fn r1cs_witness_value<const C: usize, I: ConstraintInput>(&self, index: usize) -> F {
        let trace_len = self.bytecode.v_read_write[0].len();
        I::from_index::<C>(index / trace_len).get_poly_ref(self)[index % trace_len]
    }

    pub fn commit<PCS: CommitmentScheme<Field = F>>(
        &self,
        pcs_setup: &PCS::Setup,
    ) -> JoltCommitments<PCS> {
        let mut commitments: JoltCommitments<PCS> = JoltCommitments::default();

        let trace_polys = self.read_write_values();
        let mut trace_comitments =
            PCS::batch_commit_polys_ref(&trace_polys, pcs_setup, BatchType::Big);
        commitments
            .read_write_values_mut()
            .zip(trace_comitments.into_iter())
            .for_each(|(dest, src)| *dest = src);

        let bytecode_t_final_commitment = PCS::commit(&self.bytecode.t_final, pcs_setup);
        let (memory_v_final_commitment, memory_t_final_commitment) = rayon::join(
            || PCS::commit(&self.read_write_memory.v_final, pcs_setup),
            || PCS::commit(&self.read_write_memory.t_final, pcs_setup),
        );
        let instruction_final_commitment = PCS::batch_commit_polys(
            &self.instruction_lookups.final_cts,
            pcs_setup,
            BatchType::Big,
        );

        commitments
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
            BytecodePolynomials::<F>::commit_shapes(max_bytecode_size, max_trace_length);
        let ram_commitment_shapes = ReadWriteMemoryPolynomials::<F>::commitment_shapes(
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

        let load_store_flags = &instruction_polynomials.instruction_flags[5..10];
        let (memory_polynomials, read_timestamps) = ReadWriteMemoryPolynomials::generate_witness(
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
