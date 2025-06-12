#![allow(clippy::type_complexity)]
#![allow(dead_code)]

use crate::field::JoltField;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::opening_proof::{
    ProverOpeningAccumulator, ReducedOpeningProof, VerifierOpeningAccumulator,
};
use crate::r1cs::constraints::R1CSConstraints;
use crate::r1cs::spartan::{self, UniformSpartanProof};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::rv_trace::{MemoryLayout, NUM_CIRCUIT_FLAGS};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};
use strum::EnumCount;
use timestamp_range_check::TimestampRangeCheckStuff;

use self::bytecode::{BytecodePreprocessing, BytecodeProof, BytecodeRow, BytecodeStuff};
use self::instruction_lookups::{
    InstructionLookupStuff, InstructionLookupsPreprocessing, InstructionLookupsProof,
};
use self::read_write_memory::{
    ReadWriteMemoryPolynomials, ReadWriteMemoryPreprocessing, ReadWriteMemoryProof,
    ReadWriteMemoryStuff,
};
use crate::join_conditional;
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
    Initializable, MemoryCheckingProver, MemoryCheckingVerifier, StructuredPolynomialData,
};
use crate::msm::icicle;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::r1cs::inputs::{ConstraintInput, R1CSPolynomials, R1CSProof, R1CSStuff};
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::{AppendToTranscript, Transcript};
use common::{
    constants::MEMORY_OPS_PER_INSTRUCTION,
    rv_trace::{ELFInstruction, JoltDevice, MemoryOp},
};

use super::instruction::lb::LBInstruction;
use super::instruction::lbu::LBUInstruction;
use super::instruction::lh::LHInstruction;
use super::instruction::lhu::LHUInstruction;
use super::instruction::sb::SBInstruction;
use super::instruction::sh::SHInstruction;
use super::instruction::JoltInstructionSet;

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltVerifierPreprocessing<const C: usize, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub generators: PCS::Setup,
    pub instruction_lookups: InstructionLookupsPreprocessing<C, F>,
    pub bytecode: BytecodePreprocessing<F>,
    pub read_write_memory: ReadWriteMemoryPreprocessing,
    pub memory_layout: MemoryLayout,
}

impl<const C: usize, F, PCS, ProofTranscript> JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>
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
pub struct JoltProverPreprocessing<const C: usize, F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub shared: JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
    field: F::SmallValueLookupTables,
}

impl<const C: usize, F, PCS, ProofTranscript> JoltProverPreprocessing<C, F, PCS, ProofTranscript>
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

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct JoltTraceStep<InstructionSet: JoltInstructionSet> {
    pub instruction_lookup: Option<InstructionSet>,
    pub bytecode_row: BytecodeRow,
    pub memory_ops: [MemoryOp; MEMORY_OPS_PER_INSTRUCTION],
    pub circuit_flags: [bool; NUM_CIRCUIT_FLAGS],
}

pub struct ProverDebugInfo<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    pub(crate) transcript: ProofTranscript,
    pub(crate) opening_accumulator: ProverOpeningAccumulator<F, ProofTranscript>,
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
                MemoryOp::noop_read(),  // RAM
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
pub struct JoltProof<
    const C: usize,
    const M: usize,
    I,
    F,
    PCS,
    InstructionSet,
    Subtables,
    ProofTranscript,
> where
    I: ConstraintInput,
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    InstructionSet: JoltInstructionSet,
    Subtables: JoltSubtableSet<F>,
    ProofTranscript: Transcript,
{
    pub trace_length: usize,
    pub bytecode: BytecodeProof<F, PCS, ProofTranscript>,
    pub read_write_memory: ReadWriteMemoryProof<F, PCS, ProofTranscript>,
    pub instruction_lookups:
        InstructionLookupsProof<C, M, F, PCS, InstructionSet, Subtables, ProofTranscript>,
    pub r1cs: UniformSpartanProof<C, I, F, ProofTranscript>,
    pub opening_proof: ReducedOpeningProof<F, PCS, ProofTranscript>,
}

#[derive(Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct JoltStuff<T: CanonicalSerialize + CanonicalDeserialize + Sync> {
    pub(crate) bytecode: BytecodeStuff<T>,
    pub(crate) read_write_memory: ReadWriteMemoryStuff<T>,
    pub(crate) instruction_lookups: InstructionLookupStuff<T>,
    pub(crate) timestamp_range_check: TimestampRangeCheckStuff<T>,
    pub(crate) r1cs: R1CSStuff<T>,
}

impl<T: CanonicalSerialize + CanonicalDeserialize + Sync> StructuredPolynomialData<T>
    for JoltStuff<T>
{
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

impl<
        const C: usize,
        T: CanonicalSerialize + CanonicalDeserialize + Default + Sync,
        PCS: CommitmentScheme<ProofTranscript>,
        ProofTranscript: Transcript,
    > Initializable<T, JoltVerifierPreprocessing<C, PCS::Field, PCS, ProofTranscript>>
    for JoltStuff<T>
{
    fn initialize(
        preprocessing: &JoltVerifierPreprocessing<C, PCS::Field, PCS, ProofTranscript>,
    ) -> Self {
        Self {
            bytecode: BytecodeStuff::initialize(&preprocessing.bytecode),
            read_write_memory: ReadWriteMemoryStuff::initialize(&preprocessing.read_write_memory),
            instruction_lookups: InstructionLookupStuff::initialize(
                &preprocessing.instruction_lookups,
            ),
            timestamp_range_check: TimestampRangeCheckStuff::initialize(
                &crate::lasso::memory_checking::NoPreprocessing,
            ),
            r1cs: R1CSStuff::initialize(&C),
        }
    }
}

impl<F: JoltField> JoltPolynomials<F> {
    #[tracing::instrument(skip_all, name = "JoltPolynomials::commit")]
    pub fn commit<const C: usize, PCS, ProofTranscript>(
        &self,
        preprocessing: &JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
    ) -> JoltCommitments<PCS, ProofTranscript>
    where
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
    {
        let span = tracing::span!(tracing::Level::INFO, "commit::initialize");
        let _guard = span.enter();
        let mut commitments = JoltCommitments::<PCS, ProofTranscript>::initialize(preprocessing);
        drop(_guard);
        drop(span);

        let trace_polys = self.read_write_values();
        let trace_commitments = PCS::batch_commit(&trace_polys, &preprocessing.generators);

        commitments
            .read_write_values_mut()
            .into_iter()
            .zip(trace_commitments.into_iter())
            .for_each(|(dest, src)| *dest = src);

        let span = tracing::span!(tracing::Level::INFO, "commit::t_final");
        let _guard = span.enter();
        commitments.bytecode.t_final =
            PCS::commit(&self.bytecode.t_final, &preprocessing.generators);
        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "commit::read_write_memory");
        let _guard = span.enter();
        (
            commitments.read_write_memory.v_final,
            commitments.read_write_memory.t_final,
        ) = join_conditional!(
            || PCS::commit(&self.read_write_memory.v_final, &preprocessing.generators),
            || PCS::commit(&self.read_write_memory.t_final, &preprocessing.generators)
        );
        commitments.instruction_lookups.final_cts = PCS::batch_commit(
            &self.instruction_lookups.final_cts,
            &preprocessing.generators,
        );
        drop(_guard);
        drop(span);

        commitments
    }
}

pub trait Jolt<F, PCS, const C: usize, const M: usize, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    type InstructionSet: JoltInstructionSet;
    type Subtables: JoltSubtableSet<F>;
    type Constraints: R1CSConstraints<C, F>;

    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    fn verifier_preprocess(
        bytecode: Vec<ELFInstruction>,
        memory_layout: MemoryLayout,
        memory_init: Vec<(u64, u8)>,
        max_bytecode_size: usize,
        max_memory_size: usize,
        max_trace_length: usize,
    ) -> JoltVerifierPreprocessing<C, F, PCS, ProofTranscript> {
        icicle::icicle_init();

        let instruction_lookups_preprocessing = InstructionLookupsPreprocessing::preprocess::<
            M,
            Self::InstructionSet,
            Self::Subtables,
        >();

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
                tracer::RV32IM::SH => SHInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::SB => SBInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::LBU => LBUInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::LHU => LHUInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::LB => LBInstruction::<32>::virtual_sequence(instruction),
                tracer::RV32IM::LH => LHInstruction::<32>::virtual_sequence(instruction),
                _ => vec![instruction],
            })
            .map(|instruction| BytecodeRow::from_instruction::<Self::InstructionSet>(&instruction))
            .collect();
        let bytecode_preprocessing = BytecodePreprocessing::<F>::preprocess(bytecode_rows);

        let max_poly_len: usize = [
            (max_bytecode_size + 1).next_power_of_two(), // Account for no-op prepended to bytecode
            max_trace_length.next_power_of_two(),
            max_memory_size.next_power_of_two(),
            M,
        ]
        .into_iter()
        .max()
        .unwrap();
        let generators = PCS::setup(max_poly_len);

        JoltVerifierPreprocessing {
            generators,
            memory_layout,
            instruction_lookups: instruction_lookups_preprocessing,
            bytecode: bytecode_preprocessing,
            read_write_memory: read_write_memory_preprocessing,
        }
    }

    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    fn prover_preprocess(
        bytecode: Vec<ELFInstruction>,
        memory_layout: MemoryLayout,
        memory_init: Vec<(u64, u8)>,
        max_bytecode_size: usize,
        max_memory_size: usize,
        max_trace_length: usize,
    ) -> JoltProverPreprocessing<C, F, PCS, ProofTranscript> {
        let small_value_lookup_tables = F::compute_lookup_tables();
        F::initialize_lookup_tables(small_value_lookup_tables.clone());
        let shared = Self::verifier_preprocess(
            bytecode,
            memory_layout,
            memory_init,
            max_bytecode_size,
            max_memory_size,
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
        mut trace: Vec<JoltTraceStep<Self::InstructionSet>>,
        mut preprocessing: JoltProverPreprocessing<C, F, PCS, ProofTranscript>,
    ) -> (
        JoltProof<
            C,
            M,
            <Self::Constraints as R1CSConstraints<C, F>>::Inputs,
            F,
            PCS,
            Self::InstructionSet,
            Self::Subtables,
            ProofTranscript,
        >,
        JoltCommitments<PCS, ProofTranscript>,
        JoltDevice,
        Option<ProverDebugInfo<F, ProofTranscript>>,
    ) {
        icicle::icicle_init();
        let trace_length = trace.len();
        let padded_trace_length = trace_length.next_power_of_two();
        let srs_size = PCS::srs_size(&preprocessing.shared.generators);
        let padded_log2 = padded_trace_length.log_2();
        let srs_log2 = srs_size.log_2();

        println!(
            "Trace length: {trace_length} (2^{})",
            trace_length.next_power_of_two().log_2()
        );

        if padded_trace_length > srs_size {
            panic!(
                "Padded trace length {padded_trace_length} (2^{padded_log2}) exceeds SRS size {srs_size} (2^{srs_log2}). Consider increasing the max_trace_length."
            );
        }

        F::initialize_lookup_tables(std::mem::take(&mut preprocessing.field));

        // TODO(moodlezoup): Truncate generators

        // TODO(JP): Drop padding on number of steps
        JoltTraceStep::pad(&mut trace);

        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        Self::fiat_shamir_preamble(
            &mut transcript,
            &program_io,
            &program_io.memory_layout,
            trace_length,
        );

        let instruction_polynomials =
            InstructionLookupsProof::<
                C,
                M,
                F,
                PCS,
                Self::InstructionSet,
                Self::Subtables,
                ProofTranscript,
            >::generate_witness(&preprocessing.shared.instruction_lookups, &trace);

        let memory_polynomials = ReadWriteMemoryPolynomials::generate_witness(
            &program_io,
            &preprocessing.shared.read_write_memory,
            &trace,
        );

        let (bytecode_polynomials, range_check_polys) = rayon::join(
            || {
                BytecodeProof::<F, PCS, ProofTranscript>::generate_witness(
                    &preprocessing.shared.bytecode,
                    &mut trace,
                )
            },
            || {
                TimestampValidityProof::<F, PCS, ProofTranscript>::generate_witness(
                    &memory_polynomials,
                )
            },
        );

        let r1cs_builder = Self::Constraints::construct_constraints(
            padded_trace_length,
            program_io.memory_layout.input_start,
        );
        let spartan_key = spartan::UniformSpartanProof::<
            C,
            <Self::Constraints as R1CSConstraints<C, F>>::Inputs,
            F,
            ProofTranscript,
        >::setup(&r1cs_builder, padded_trace_length);

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

        let jolt_commitments =
            jolt_polynomials.commit::<C, PCS, ProofTranscript>(&preprocessing.shared);

        transcript.append_scalar(&spartan_key.vk_digest);

        jolt_commitments
            .read_write_values()
            .iter()
            .for_each(|value| value.append_to_transcript(&mut transcript));
        jolt_commitments
            .init_final_values()
            .iter()
            .for_each(|value| value.append_to_transcript(&mut transcript));

        let mut opening_accumulator: ProverOpeningAccumulator<F, ProofTranscript> =
            ProverOpeningAccumulator::new();

        let bytecode_proof = BytecodeProof::prove_memory_checking(
            &preprocessing.shared.generators,
            &preprocessing.shared.bytecode,
            &jolt_polynomials.bytecode,
            &jolt_polynomials,
            &mut opening_accumulator,
            &mut transcript,
        );

        let instruction_proof = InstructionLookupsProof::prove(
            &preprocessing.shared.generators,
            &mut jolt_polynomials,
            &preprocessing.shared.instruction_lookups,
            &mut opening_accumulator,
            &mut transcript,
        );

        let memory_proof = ReadWriteMemoryProof::prove(
            &preprocessing.shared.generators,
            &preprocessing.shared.read_write_memory,
            &jolt_polynomials,
            &program_io,
            &mut opening_accumulator,
            &mut transcript,
        );

        let spartan_proof = UniformSpartanProof::<
            C,
            <Self::Constraints as R1CSConstraints<C, F>>::Inputs,
            F,
            ProofTranscript,
        >::prove::<PCS>(
            &r1cs_builder,
            &spartan_key,
            &jolt_polynomials,
            &mut opening_accumulator,
            &mut transcript,
        )
        .expect("r1cs proof failed");

        // Batch-prove all openings
        let opening_proof = opening_accumulator
            .reduce_and_prove::<PCS>(&preprocessing.shared.generators, &mut transcript);

        drop_in_background_thread(jolt_polynomials);

        let jolt_proof = JoltProof {
            trace_length,
            bytecode: bytecode_proof,
            read_write_memory: memory_proof,
            instruction_lookups: instruction_proof,
            r1cs: spartan_proof,
            opening_proof,
        };

        #[cfg(test)]
        let debug_info = Some(ProverDebugInfo {
            transcript,
            opening_accumulator,
        });
        #[cfg(not(test))]
        let debug_info = None;
        (jolt_proof, jolt_commitments, program_io, debug_info)
    }

    #[tracing::instrument(skip_all)]
    fn verify(
        mut preprocessing: JoltVerifierPreprocessing<C, F, PCS, ProofTranscript>,
        proof: JoltProof<
            C,
            M,
            <Self::Constraints as R1CSConstraints<C, F>>::Inputs,
            F,
            PCS,
            Self::InstructionSet,
            Self::Subtables,
            ProofTranscript,
        >,
        commitments: JoltCommitments<PCS, ProofTranscript>,
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
        let memory_start = preprocessing.memory_layout.input_start;
        let r1cs_builder =
            Self::Constraints::construct_constraints(padded_trace_length, memory_start);
        let spartan_key = spartan::UniformSpartanProof::<C, _, F, ProofTranscript>::setup(
            &r1cs_builder,
            padded_trace_length,
        );
        transcript.append_scalar(&spartan_key.vk_digest);

        let r1cs_proof = R1CSProof {
            key: spartan_key,
            proof: proof.r1cs,
            _marker: PhantomData,
        };

        commitments
            .read_write_values()
            .iter()
            .for_each(|value| value.append_to_transcript(&mut transcript));
        commitments
            .init_final_values()
            .iter()
            .for_each(|value| value.append_to_transcript(&mut transcript));

        Self::verify_bytecode(
            &preprocessing.bytecode,
            &preprocessing.generators,
            proof.bytecode,
            &commitments,
            &mut opening_accumulator,
            &mut transcript,
        )?;
        Self::verify_instruction_lookups(
            &preprocessing.instruction_lookups,
            &preprocessing.generators,
            proof.instruction_lookups,
            &commitments,
            &mut opening_accumulator,
            &mut transcript,
        )?;
        Self::verify_memory(
            &mut preprocessing.read_write_memory,
            &preprocessing.generators,
            &preprocessing.memory_layout,
            proof.read_write_memory,
            &commitments,
            program_io,
            &mut opening_accumulator,
            &mut transcript,
        )?;
        Self::verify_r1cs(
            r1cs_proof,
            &commitments,
            &mut opening_accumulator,
            &mut transcript,
        )?;

        // Batch-verify all openings
        opening_accumulator.reduce_and_verify(
            &preprocessing.generators,
            &proof.opening_proof,
            &mut transcript,
        )?;

        Ok(())
    }

    #[tracing::instrument(skip_all)]
    fn verify_instruction_lookups<'a>(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        generators: &PCS::Setup,
        proof: InstructionLookupsProof<
            C,
            M,
            F,
            PCS,
            Self::InstructionSet,
            Self::Subtables,
            ProofTranscript,
        >,
        commitments: &'a JoltCommitments<PCS, ProofTranscript>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        InstructionLookupsProof::verify(
            preprocessing,
            generators,
            proof,
            commitments,
            opening_accumulator,
            transcript,
        )
    }

    #[tracing::instrument(skip_all)]
    fn verify_bytecode<'a>(
        preprocessing: &BytecodePreprocessing<F>,
        generators: &PCS::Setup,
        proof: BytecodeProof<F, PCS, ProofTranscript>,
        commitments: &'a JoltCommitments<PCS, ProofTranscript>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        BytecodeProof::verify_memory_checking(
            preprocessing,
            generators,
            proof,
            &commitments.bytecode,
            commitments,
            opening_accumulator,
            transcript,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all)]
    fn verify_memory<'a>(
        preprocessing: &mut ReadWriteMemoryPreprocessing,
        generators: &PCS::Setup,
        memory_layout: &MemoryLayout,
        proof: ReadWriteMemoryProof<F, PCS, ProofTranscript>,
        commitment: &'a JoltCommitments<PCS, ProofTranscript>,
        program_io: JoltDevice,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        assert!(program_io.inputs.len() <= memory_layout.max_input_size as usize);
        assert!(program_io.outputs.len() <= memory_layout.max_output_size as usize);
        // pair the memory layout with the program io from the proof
        preprocessing.program_io = Some(JoltDevice {
            inputs: program_io.inputs,
            outputs: program_io.outputs,
            panic: program_io.panic,
            memory_layout: memory_layout.clone(),
        });

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
        proof: R1CSProof<
            C,
            <Self::Constraints as R1CSConstraints<C, F>>::Inputs,
            F,
            ProofTranscript,
        >,
        commitments: &'a JoltCommitments<PCS, ProofTranscript>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        proof
            .verify(commitments, opening_accumulator, transcript)
            .map_err(|e| ProofVerifyError::SpartanError(e.to_string()))
    }

    fn fiat_shamir_preamble(
        transcript: &mut ProofTranscript,
        program_io: &JoltDevice,
        memory_layout: &MemoryLayout,
        trace_length: usize,
    ) {
        transcript.append_u64(trace_length as u64);
        transcript.append_u64(C as u64);
        transcript.append_u64(M as u64);
        transcript.append_u64(Self::InstructionSet::COUNT as u64);
        transcript.append_u64(Self::Subtables::COUNT as u64);
        transcript.append_u64(memory_layout.max_input_size);
        transcript.append_u64(memory_layout.max_output_size);
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
