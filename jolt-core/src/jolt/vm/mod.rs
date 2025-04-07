#![allow(clippy::type_complexity)]
#![allow(dead_code)]

use std::marker::PhantomData;
use std::slice::Iter;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::log2;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use strum::EnumCount;

use bytecode::{BytecodeOracle, DerivedOracle};
use common::{
    constants::MEMORY_OPS_PER_INSTRUCTION,
    rv_trace::{ELFInstruction, JoltDevice, MemoryOp},
};
use common::rv_trace::{MemoryLayout, NUM_CIRCUIT_FLAGS};
use instruction_lookups::InstructionLookupOracle;
use read_write_memory::ReadWriteMemoryOracle;
use timestamp_range_check::TimestampRangeCheckStuff;

use crate::field::JoltField;
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
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::opening_proof::{
    ProverOpeningAccumulator, ReducedOpeningProof, VerifierOpeningAccumulator,
};
use crate::r1cs::builder::CombinedUniformBuilder;
use crate::r1cs::constraints::R1CSConstraints;
use crate::r1cs::inputs::{
    AuxVariableStuff, ConstraintInput, R1CSPolynomials, R1CSProof, R1CSStuff,
};
use crate::r1cs::spartan::{self, UniformSpartanProof};
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::utils::streaming::Oracle;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::transcript::{AppendToTranscript, Transcript};

use super::instruction::JoltInstructionSet;
use super::instruction::lb::LBInstruction;
use super::instruction::lbu::LBUInstruction;
use super::instruction::lh::LHInstruction;
use super::instruction::lhu::LHUInstruction;
use super::instruction::sb::SBInstruction;
use super::instruction::sh::SHInstruction;

use self::bytecode::{BytecodePreprocessing, BytecodeProof, BytecodeRow, BytecodeStuff};
use self::instruction_lookups::{
    InstructionLookupsPreprocessing, InstructionLookupsProof, InstructionLookupStuff,
};
use self::read_write_memory::{
    ReadWriteMemoryPolynomials, ReadWriteMemoryPreprocessing, ReadWriteMemoryProof,
    ReadWriteMemoryStuff,
};

pub mod bytecode;
pub mod instruction_lookups;
pub mod read_write_memory;
pub mod rv32i_vm;
pub mod timestamp_range_check;

#[derive(Clone)]
pub struct JoltPreprocessing<const C: usize, F, PCS, ProofTranscript>
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
    field: F::SmallValueLookupTables,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct JoltTraceStep<InstructionSet: JoltInstructionSet> {
    pub instruction_lookup: Option<InstructionSet>,
    pub bytecode_row: BytecodeRow,
    pub memory_ops: [MemoryOp; MEMORY_OPS_PER_INSTRUCTION],
    pub circuit_flags: [bool; NUM_CIRCUIT_FLAGS],
}

pub struct TraceOracle<'a, InstructionSet: JoltInstructionSet> {
    pub length: usize,
    pub step: usize,
    pub trace: &'a Vec<JoltTraceStep<InstructionSet>>,
}

impl<'a, InstructionSet: JoltInstructionSet> TraceOracle<'a, InstructionSet> {
    pub fn new(trace: &'a Vec<JoltTraceStep<InstructionSet>>) -> Self {
        Self {
            length: trace.len(),
            step: 0,
            trace: &trace,
        }
    }
}

impl<'a, InstructionSet: JoltInstructionSet> Oracle for TraceOracle<'a, InstructionSet> {
    type Item = &'a [JoltTraceStep<InstructionSet>];

    // TODO (Bhargav): This should return an Option. Return None if trace exhasuted.
    fn next_shard(&mut self, shard_len: usize) -> Self::Item {
        let shard_start = self.step;
        self.step += shard_len;
        &self.trace[shard_start..self.step]
    }

    fn reset_oracle(&mut self) {
        if self.step == self.length {
            self.step = 0;
        } else {
            println!("Trace length = {}, step = {}", self.length, self.step);
            panic!("Can't reset, trace not exhausted.");
        }
    }

    fn peek(&mut self) -> Self::Item {
        &self.trace[self.step..self.step + 1]
    }

    fn get_len(&self) -> usize {
        self.length
    }

    fn get_step(&self) -> usize {
        self.step
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
    pub program_io: JoltDevice,
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

pub struct JoltOracle<'a, F: JoltField, InstructionSet: JoltInstructionSet> {
    pub trace_oracle: TraceOracle<'a, InstructionSet>,
    pub bytecode_oracle: BytecodeOracle<'a, F, InstructionSet>,
    pub instruction_lookups_oracle: InstructionLookupOracle<'a, F, InstructionSet>,
    pub read_write_memory_oracle: ReadWriteMemoryOracle<'a, F, InstructionSet>,
    pub func: Box<
        dyn Fn(
                &[JoltTraceStep<InstructionSet>],
                BytecodeStuff<MultilinearPolynomial<F>>,
                InstructionLookupStuff<MultilinearPolynomial<F>>,
                ReadWriteMemoryStuff<MultilinearPolynomial<F>>,
            ) -> JoltStuff<MultilinearPolynomial<F>>
            + 'a,
    >,
}

impl<'a, F: JoltField, InstructionSet: JoltInstructionSet> JoltOracle<'a, F, InstructionSet> {
    pub fn new<const C: usize, const M: usize, PCS, ProofTranscript, CI: ConstraintInput>(
        preprocessing: &'a JoltPreprocessing<C, F, PCS, ProofTranscript>,
        program_io: &'a JoltDevice,
        r1cs_builder: &'a CombinedUniformBuilder<C, F, CI>,
        trace: &'a Vec<JoltTraceStep<InstructionSet>>,
    ) -> Self
    where
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
    {
        let mut trace_oracle = TraceOracle::new(trace);

        let mut bytecode_oracle = BytecodeOracle::new(&preprocessing.bytecode, trace);

        let mut instruction_lookups_oracle =
            InstructionLookupOracle::new::<C, M>(&preprocessing.instruction_lookups, trace);

        // let program_io_clone = program_io.clone();
        let mut read_write_memory_oracle = ReadWriteMemoryOracle::new(
            &preprocessing.read_write_memory,
            program_io,
            trace,
            PhantomData::<F>,
        );

        let polynomial_stream = Box::new(
            |shard: &[JoltTraceStep<InstructionSet>],
             bytecode: BytecodeStuff<MultilinearPolynomial<F>>,
             instruction_lookups: InstructionLookupStuff<MultilinearPolynomial<F>>,
             read_write_memory: ReadWriteMemoryStuff<MultilinearPolynomial<F>>| {
                let shard_len = shard.len();
                let mut chunks_x = vec![vec![0u8; shard_len]; C];
                let mut chunks_y = vec![vec![0u8; shard_len]; C];
                let mut circuit_flags = vec![vec![0u8; shard_len]; NUM_CIRCUIT_FLAGS];
                let log_M = log2(M) as usize;

                for i in 0..shard_len {
                    let step = &shard[i];
                    if let Some(instr) = &step.instruction_lookup {
                        let (x, y) = instr.operand_chunks(C, log_M);
                        for j in 0..C {
                            chunks_x[j][i] = x[j];
                            chunks_y[j][i] = y[j];
                        }
                    }

                    for j in 0..NUM_CIRCUIT_FLAGS {
                        if step.circuit_flags[j] {
                            circuit_flags[j][i] = 1;
                        }
                    }
                }

                let r1cs = R1CSStuff {
                    chunks_x: chunks_x
                        .into_iter()
                        .map(MultilinearPolynomial::from)
                        .collect(),
                    chunks_y: chunks_y
                        .into_iter()
                        .map(MultilinearPolynomial::from)
                        .collect(),
                    circuit_flags: circuit_flags
                        .into_iter()
                        .map(MultilinearPolynomial::from)
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap(),
                    aux: AuxVariableStuff::initialize(&C),
                };

                let mut jolt_polys = JoltStuff {
                    bytecode,
                    instruction_lookups,
                    read_write_memory,
                    timestamp_range_check: Default::default(),
                    r1cs,
                };

                r1cs_builder.streaming_compute_aux(&mut jolt_polys, shard_len);

                jolt_polys
            },
        );

        JoltOracle {
            trace_oracle,
            bytecode_oracle,
            instruction_lookups_oracle,
            read_write_memory_oracle,
            func: Box::new(polynomial_stream),
        }
    }
}

impl<'a, F: JoltField, InstructionSet: JoltInstructionSet> Oracle
    for JoltOracle<'a, F, InstructionSet>
{
    type Item = JoltStuff<MultilinearPolynomial<F>>;

    // TODO (Bhargav): This should return an Option. Return None if trace exhasuted.
    fn next_shard(&mut self, shard_len: usize) -> Self::Item {
        (self.func)(
            self.trace_oracle.next_shard(shard_len),
            self.bytecode_oracle.next_shard(shard_len),
            self.instruction_lookups_oracle.next_shard(shard_len),
            self.read_write_memory_oracle.next_shard(shard_len),
        )
    }

    fn reset_oracle(&mut self) {
        self.trace_oracle.reset_oracle();
        self.bytecode_oracle.reset_oracle();
        self.instruction_lookups_oracle.reset_oracle();
        self.read_write_memory_oracle.reset_oracle();
    }

    fn peek(&mut self) -> Self::Item {
        (self.func)(
            self.trace_oracle.peek(),
            self.bytecode_oracle.peek(),
            self.instruction_lookups_oracle.peek(),
            self.read_write_memory_oracle.peek(),
        )
    }

    // TODO: Add asserts to check that get_len() of all oracles returns the same value?
    fn get_len(&self) -> usize {
        self.trace_oracle.get_len()
    }

    // TODO: Add asserts to check that get_step() of all oracles returns the same value?
    fn get_step(&self) -> usize {
        self.trace_oracle.get_step()
    }
}

//  impl<'a, F: JoltField, InstructionSet: JoltInstructionSet> JoltOracle<'a, F, InstructionSet> {
//     pub fn new<const C: usize, const M: usize, PCS, ProofTranscript>(
//         preprocessing: &'a JoltPreprocessing<C, F, PCS, ProofTranscript>,
//         trace: &'a Vec<JoltTraceStep<InstructionSet>>,
//     ) -> Self
//     where
//         PCS: CommitmentScheme<ProofTranscript, Field = F>,
//         ProofTranscript: Transcript,
//     {
//         let mut trace_oracle = TraceOracle::new(trace);

//         let max_trace_address = trace
//             .into_iter()
//             .map(|step| match step.memory_ops[RAM] {
//                 MemoryOp::Read(a) => remap_address(a, &program_io.memory_layout),
//                 MemoryOp::Write(a, _) => remap_address(a, &program_io.memory_layout),
//             })
//             .max()
//             .unwrap();

//         let memory_size = max_trace_address.next_power_of_two() as usize;
//         let mut v_init: Vec<u32> = vec![0; memory_size];

//         // Copy bytecode
//         let mut v_init_index = memory_address_to_witness_index(
//             preprocessing.min_bytecode_address,
//             &program_io.memory_layout,
//         );

//         for word in preprocessing.bytecode_words.iter() {
//             v_init[v_init_index] = *word;
//             v_init_index += 1;
//         }
//         // Copy input bytes
//         v_init_index = memory_address_to_witness_index(
//             program_io.memory_layout.input_start,
//             &program_io.memory_layout,
//         );

//         // Convert input bytes into words and populate `v_init`
//         for chunk in program_io.inputs.chunks(4) {
//             let mut word = [0u8; 4];
//             for (i, byte) in chunk.iter().enumerate() {
//                 word[i] = *byte;
//             }
//             let word = u32::from_le_bytes(word);
//             v_init[v_init_index] = word;
//             v_init_index += 1;
//         }
//         let v_final = v_init;

//         let polynomial_stream = (|step: JoltTraceStep<InstructionSet>| {
//             let virtual_address = preprocessing
//                 .bytecode
//                 .virtual_address_map
//                 .get(&(
//                     step.bytecode_row.address,
//                     step.bytecode_row.virtual_sequence_remaining.unwrap_or(0),
//                 ))
//                 .unwrap();
//             let a_read_write = *virtual_address as u64;

//             let address = F::from_u64(step.bytecode_row.address as u64);
//             let bitflags = F::from_u64(step.bytecode_row.bitflags);
//             let rd = F::from_u8(step.bytecode_row.rd);
//             let rs1 = F::from_u8(step.bytecode_row.rs1);
//             let rs2 = F::from_u8(step.bytecode_row.rs2);
//             let imm = F::from_i64(step.bytecode_row.imm);

//             let v_read_write = [address, bitflags, rd, rs1, rs2, imm];

//             let bytecode = BytecodeStuff {
//                 a_read_write: F::from_u64(a_read_write),
//                 v_read_write,
//                 // These are dummy values since they are not required for Twist + Shout.
//                 t_read: -F::one(),
//                 t_final: -F::one(),
//                 a_init_final: None,
//                 v_init_final: None,
//             };

//             let chunked_indices: Vec<u16> = if let Some(instr) = &step.instruction_lookup {
//                 instr
//                     .to_indices(C, M.log_2())
//                     .iter()
//                     .map(|i| *i as u16)
//                     .collect()
//             } else {
//                 vec![0; C]
//             };
//             let mut subtable_lookup_indices: Vec<u16> = Vec::with_capacity(C);
//             for i in 0..C {
//                 subtable_lookup_indices.push(chunked_indices[i]);
//             }

//             //Computing dim
//             let dim: Vec<F> = subtable_lookup_indices
//                 .clone()
//                 .into_par_iter()
//                 .map(F::from_u16)
//                 .collect();

//             //Computing E_polys
//             let E_polynomials: Vec<F> = (0..preprocessing.instruction_lookups.num_memories)
//                 .into_par_iter()
//                 .map(|memory_index| {
//                     let dim_index =
//                         preprocessing.instruction_lookups.memory_to_dimension_index[memory_index];
//                     let subtable_index =
//                         preprocessing.instruction_lookups.memory_to_subtable_index[memory_index];
//                     let access_sequence = subtable_lookup_indices[dim_index];
//                     let mut subtable_lookups: u32 = 0;
//                     if let Some(instr) = &step.instruction_lookup {
//                         let memories_used = &preprocessing
//                             .instruction_lookups
//                             .instruction_to_memory_indices[InstructionSet::enum_index(instr)];
//                         if memories_used.contains(&memory_index) {
//                             let memory_address = access_sequence as usize;
//                             debug_assert!(memory_address < M);
//                             subtable_lookups = preprocessing
//                                 .instruction_lookups
//                                 .materialized_subtables[subtable_index][memory_address];
//                         }
//                     }
//                     F::from_u32(subtable_lookups)
//                 })
//                 .collect();

//             // Computing instruction_flags
//             let mut instruction_flag_bitvectors: Vec<F> = vec![
//                 F::zero();
//                 preprocessing
//                     .instruction_lookups
//                     .instruction_to_memory_indices
//                     .len()
//             ];
//             if let Some(instr) = &step.instruction_lookup {
//                 instruction_flag_bitvectors[InstructionSet::enum_index(instr)] = F::one();
//             }

//             // Computing instruction lookups
//             let lookup_outputs = if let Some(instr) = &step.instruction_lookup {
//                 F::from_u32(instr.lookup_entry() as u32)
//             } else {
//                 F::zero()
//             };

//             let instruction_lookup = InstructionLookupStuff {
//                 dim,
//                 E_polys: E_polynomials,
//                 instruction_flags: instruction_flag_bitvectors,
//                 lookup_outputs,
//                 // These are dummy values since they are not required for Twist + Shout.
//                 read_cts: vec![F::zero()],
//                 final_cts: vec![F::zero()],
//                 a_init_final: None,
//                 v_init_final: None,
//             };

//             let a_ram;
//             let v_read_rd;
//             let v_read_rs1;
//             let v_read_rs2;
//             let v_read_ram;
//             let v_write_rd;
//             let v_write_ram;

//             match step.memory_ops[RS1] {
//                 MemoryOp::Read(a) => {
//                     assert!(a < REGISTER_COUNT);
//                     let a = a as usize;
//                     let v = v_final[a];

//                     v_read_rs1 = v;
//                 }
//                 MemoryOp::Write(a, v) => {
//                     panic!("Unexpected rs1 MemoryOp::Write({}, {})", a, v);
//                 }
//             };

//             match step.memory_ops[RS2] {
//                 MemoryOp::Read(a) => {
//                     assert!(a < REGISTER_COUNT);
//                     let a = a as usize;
//                     let v = v_final[a];

//                     v_read_rs2 = v;
//                 }
//                 MemoryOp::Write(a, v) => {
//                     panic!("Unexpected rs2 MemoryOp::Write({}, {})", a, v)
//                 }
//             };

//             match step.memory_ops[RD] {
//                 MemoryOp::Read(a) => {
//                     panic!("Unexpected rd MemoryOp::Read({})", a)
//                 }
//                 MemoryOp::Write(a, v_new) => {
//                     assert!(a < REGISTER_COUNT);
//                     let a = a as usize;
//                     let v_old = v_final[a];

//                     v_read_rd = v_old;
//                     v_write_rd = v_new as u32;
//                     v_final[a] = v_new as u32;
//                 }
//             };

//             match step.memory_ops[RAM] {
//                 MemoryOp::Read(a) => {
//                     debug_assert!(a % 4 == 0);
//                     let remapped_a = remap_address(a, &program_io.memory_layout) as usize;
//                     let v = v_final[remapped_a];

//                     a_ram = remapped_a as u32;
//                     v_read_ram = v;
//                     v_write_ram = v;
//                 }
//                 MemoryOp::Write(a, v_new) => {
//                     debug_assert!(a % 4 == 0);
//                     let remapped_a = remap_address(a, &program_io.memory_layout) as usize;
//                     let v_old = v_final[remapped_a];

//                     a_ram = remapped_a as u32;
//                     v_read_ram = v_old;
//                     v_write_ram = v_new as u32;
//                     v_final[remapped_a] = v_new as u32;
//                 }
//             }

//             ReadWriteMemoryStuff {
//                 a_ram: F::from_u32(a_ram),
//                 v_read_rd: F::from_u32(v_read_rd),
//                 v_read_rs1: F::from_u32(v_read_rs1),
//                 v_read_rs2: F::from_u32(v_read_rs2),
//                 v_read_ram: F::from_u32(v_read_ram),
//                 v_write_rd: F::from_u32(v_write_rd),
//                 v_write_ram: F::from_u32(v_write_ram),
//                 // These are dummy values since they are not required for Twist + Shout.
//                 v_final: F::zero(),
//                 t_read_rd: F::zero(),
//                 t_read_rs1: F::zero(),
//                 t_read_rs2: F::zero(),
//                 t_read_ram: F::zero(),
//                 t_final: F::zero(),
//                 a_init_final: None,
//                 v_init: None,
//                 identity: None,
//             }
//         });

//         // BytecodeOracle {
//         //     trace_oracle,
//         //     func: Box::new(polynomial_stream),
//         // }
//     }
// }

//TODO: Implment StreamingOracle for StreamingJoltStuff.

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

// pub type StreamingJoltPolynomials<F: JoltField> = JoltStuff<StreamingPolynomial<JoltTraceStep<JoltInstructionSet>, F>>;
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
    > Initializable<T, JoltPreprocessing<C, PCS::Field, PCS, ProofTranscript>> for JoltStuff<T>
{
    fn initialize(preprocessing: &JoltPreprocessing<C, PCS::Field, PCS, ProofTranscript>) -> Self {
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
        preprocessing: &JoltPreprocessing<C, F, PCS, ProofTranscript>,
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
    fn preprocess(
        bytecode: Vec<ELFInstruction>,
        memory_layout: MemoryLayout,
        memory_init: Vec<(u64, u8)>,
        max_bytecode_size: usize,
        max_memory_address: usize,
        max_trace_length: usize,
    ) -> JoltPreprocessing<C, F, PCS, ProofTranscript> {
        let small_value_lookup_tables = F::compute_lookup_tables();
        F::initialize_lookup_tables(small_value_lookup_tables.clone());
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
            max_memory_address.next_power_of_two(),
            M,
        ]
        .into_iter()
        .max()
        .unwrap();
        let generators = PCS::setup(max_poly_len);

        JoltPreprocessing {
            generators,
            memory_layout,
            instruction_lookups: instruction_lookups_preprocessing,
            bytecode: bytecode_preprocessing,
            read_write_memory: read_write_memory_preprocessing,
            field: small_value_lookup_tables,
        }
    }

    #[tracing::instrument(skip_all, name = "Jolt::prove")]
    fn prove<'a>(
        program_io: JoltDevice,
        mut trace: Vec<JoltTraceStep<Self::InstructionSet>>,
        mut preprocessing: JoltPreprocessing<C, F, PCS, ProofTranscript>,
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
        Option<ProverDebugInfo<F, ProofTranscript>>,
    ) {
        icicle::icicle_init();
        let trace_length = trace.len();
        let padded_trace_length = trace_length.next_power_of_two();
        println!("Trace length: {}", trace_length);

        F::initialize_lookup_tables(std::mem::take(&mut preprocessing.field));

        // TODO(moodlezoup): Truncate generators

        // TODO(JP): Drop padding on number of steps
        JoltTraceStep::pad(&mut trace);
        let mut trace_1 = trace.clone();

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
            >::generate_witness(&preprocessing.instruction_lookups, &trace);

        let memory_polynomials = ReadWriteMemoryPolynomials::generate_witness(
            &program_io,
            &preprocessing.read_write_memory,
            &trace,
        );

        let (bytecode_polynomials, range_check_polys) = rayon::join(
            || {
                BytecodeProof::<F, PCS, ProofTranscript>::generate_witness(
                    &preprocessing.bytecode,
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

        // TODO: Stream Polynomials

        BytecodeOracle::<F, Self::InstructionSet>::update_trace(&mut trace_1);

        let mut bytecode_oracle = BytecodeOracle::new(&preprocessing.bytecode, &trace_1);

        let mut instruction_lookups_oracle =
            InstructionLookupOracle::new::<C, M>(&preprocessing.instruction_lookups, &trace_1);

        let program_io_clone = program_io.clone();
        let mut read_write_memory_oracle = ReadWriteMemoryOracle::new(
            &preprocessing.read_write_memory,
            &program_io_clone,
            &trace_1,
            PhantomData::<F>,
        );

        let shard_len = padded_trace_length / 2;
        let no_of_shards = padded_trace_length / shard_len;

        for n in 0..no_of_shards {
            let streamed_polys = bytecode_oracle.next_shard(shard_len);
            for i in 0..shard_len {
                assert_eq!(
                    streamed_polys.a_read_write.get_coeff(i),
                    bytecode_polynomials
                        .a_read_write
                        .get_coeff(n * shard_len + i)
                );
            }

            for j in 0..6 {
                for i in 0..shard_len {
                    assert_eq!(
                        streamed_polys.v_read_write[j].get_coeff(i),
                        bytecode_polynomials.v_read_write[j].get_coeff(n * shard_len + i)
                    );
                }
            }
            println!("Streaming for bytecode_polynomials  passing for {n}th shard");
        }

        for n in 0..no_of_shards {
            let streamed_polys = instruction_lookups_oracle.next_shard(shard_len);
            for i in 0..shard_len {
                for poly_index in 0..instruction_polynomials.dim.len() {
                    assert_eq!(
                        streamed_polys.dim[poly_index].get_coeff(i),
                        instruction_polynomials.dim[poly_index].get_coeff(n * shard_len + i)
                    );
                }
                for poly_index in 0..instruction_polynomials.E_polys.len() {
                    assert_eq!(
                        streamed_polys.E_polys[poly_index].get_coeff(i),
                        instruction_polynomials.E_polys[poly_index].get_coeff(n * shard_len + i)
                    );
                }
                for poly_index in 0..instruction_polynomials.instruction_flags.len() {
                    assert_eq!(
                        streamed_polys.instruction_flags[poly_index].get_coeff(i),
                        instruction_polynomials.instruction_flags[poly_index]
                            .get_coeff(n * shard_len + i)
                    );
                }
            }
            for i in 0..shard_len {
                assert_eq!(
                    streamed_polys.lookup_outputs.get_coeff(i),
                    instruction_polynomials
                        .lookup_outputs
                        .get_coeff(n * shard_len + i)
                );
            }
            println!("Streaming for Instruction polynomials passing for {n}th shard");
        }

        // testing the streaming polynomials for memory_polynomials
        for n in 0..no_of_shards {
            let streamed_polys = read_write_memory_oracle.next_shard(shard_len);
            for i in 0..shard_len {
                assert_eq!(
                    streamed_polys.a_ram.get_coeff(i),
                    memory_polynomials.a_ram.get_coeff(n * shard_len + i)
                );
                assert_eq!(
                    streamed_polys.v_read_rs1.get_coeff(i),
                    memory_polynomials.v_read_rs1.get_coeff(n * shard_len + i)
                );
                assert_eq!(
                    streamed_polys.v_read_rs2.get_coeff(i),
                    memory_polynomials.v_read_rs2.get_coeff(n * shard_len + i)
                );
                assert_eq!(
                    streamed_polys.v_read_rd.get_coeff(i),
                    memory_polynomials.v_read_rd.get_coeff(n * shard_len + i)
                );
                assert_eq!(
                    streamed_polys.v_write_rd.get_coeff(i),
                    memory_polynomials.v_write_rd.get_coeff(n * shard_len + i)
                );
                assert_eq!(
                    streamed_polys.v_read_ram.get_coeff(i),
                    memory_polynomials.v_read_ram.get_coeff(n * shard_len + i)
                );
                assert_eq!(
                    streamed_polys.v_write_ram.get_coeff(i),
                    memory_polynomials.v_write_ram.get_coeff(n * shard_len + i)
                );
            }
            println!("Streaming for Memory polynomials passing for {n}th shard");
        }

        let mut jolt_polynomials = JoltPolynomials {
            bytecode: bytecode_polynomials,
            read_write_memory: memory_polynomials,
            timestamp_range_check: range_check_polys,
            instruction_lookups: instruction_polynomials,
            r1cs: r1cs_polynomials,
        };

        r1cs_builder.compute_aux(&mut jolt_polynomials);

        let program_io_clone = program_io.clone();
        let mut jolt_oracle = JoltOracle::new::<
            C,
            M,
            PCS,
            ProofTranscript,
            <Self::Constraints as R1CSConstraints<C, F>>::Inputs,
        >(&preprocessing, &program_io_clone, &r1cs_builder, &trace_1);

        for n in 0..no_of_shards {
            let streamed_polys = jolt_oracle.next_shard(shard_len);
            for shard in 0..shard_len {
                for i in 0..C {
                    assert_eq!(
                        streamed_polys.r1cs.chunks_x[i].get_coeff(shard),
                        jolt_polynomials.r1cs.chunks_x[i].get_coeff(n * shard_len + shard)
                    );
                    assert_eq!(
                        streamed_polys.r1cs.chunks_y[i].get_coeff(shard),
                        jolt_polynomials.r1cs.chunks_y[i].get_coeff(n * shard_len + shard)
                    );
                }
                for i in 0..NUM_CIRCUIT_FLAGS {
                    assert_eq!(
                        streamed_polys.r1cs.circuit_flags[i].get_coeff(shard),
                        jolt_polynomials.r1cs.circuit_flags[i].get_coeff(n * shard_len + shard)
                    );
                }
                for i in 0..jolt_polynomials.r1cs.aux.relevant_y_chunks.len() {
                    assert_eq!(
                        streamed_polys.r1cs.aux.relevant_y_chunks[i].get_coeff(shard),
                        jolt_polynomials.r1cs.aux.relevant_y_chunks[i]
                            .get_coeff(n * shard_len + shard)
                    );
                    assert_eq!(
                        streamed_polys.r1cs.aux.left_lookup_operand.get_coeff(shard),
                        jolt_polynomials
                            .r1cs
                            .aux
                            .left_lookup_operand
                            .get_coeff(n * shard_len + shard)
                    );
                    assert_eq!(
                        streamed_polys
                            .r1cs
                            .aux
                            .right_lookup_operand
                            .get_coeff(shard),
                        jolt_polynomials
                            .r1cs
                            .aux
                            .right_lookup_operand
                            .get_coeff(n * shard_len + shard)
                    );
                    assert_eq!(
                        streamed_polys.r1cs.aux.product.get_coeff(shard),
                        jolt_polynomials
                            .r1cs
                            .aux
                            .product
                            .get_coeff(n * shard_len + shard)
                    );
                    assert_eq!(
                        streamed_polys
                            .r1cs
                            .aux
                            .write_lookup_output_to_rd
                            .get_coeff(shard),
                        jolt_polynomials
                            .r1cs
                            .aux
                            .write_lookup_output_to_rd
                            .get_coeff(n * shard_len + shard)
                    );
                    assert_eq!(
                        streamed_polys.r1cs.aux.write_pc_to_rd.get_coeff(shard),
                        jolt_polynomials
                            .r1cs
                            .aux
                            .write_pc_to_rd
                            .get_coeff(n * shard_len + shard)
                    );
                    assert_eq!(
                        streamed_polys.r1cs.aux.next_pc_jump.get_coeff(shard),
                        jolt_polynomials
                            .r1cs
                            .aux
                            .next_pc_jump
                            .get_coeff(n * shard_len + shard)
                    );
                    assert_eq!(
                        streamed_polys.r1cs.aux.should_branch.get_coeff(shard),
                        jolt_polynomials
                            .r1cs
                            .aux
                            .should_branch
                            .get_coeff(n * shard_len + shard)
                    );
                    assert_eq!(
                        streamed_polys.r1cs.aux.next_pc.get_coeff(shard),
                        jolt_polynomials
                            .r1cs
                            .aux
                            .next_pc
                            .get_coeff(n * shard_len + shard)
                    );
                }
            }
            println!("Streaming for R1CS polynomials passing for {n}th shard");
        }

        jolt_oracle.reset_oracle();

        let jolt_commitments = jolt_polynomials.commit::<C, PCS, ProofTranscript>(&preprocessing);

        transcript.append_scalar(&spartan_key.vk_digest);

        // for i in 0..10 {
        //     println!(
        //         "bytecode_oracle's {i}-th address = {}",
        //         bytecode_oracle.next_eval().a_read_write
        //     );
        // }

        // let program_io_clone = program_io.clone();
        // let mut jolt_oracle = JoltOracle::new::<C, M, PCS, ProofTranscript>(
        //     &preprocessing,
        //     &program_io_clone,
        //     &trace_1,
        // );

        // for i in 0..trace_length {
        //     let eval = jolt_oracle.next_eval();
        //     assert_eq!(
        //         eval.bytecode.a_read_write,
        //         jolt_polynomials.bytecode.a_read_write.get_coeff(i)
        //     );

        //     for j in 0..6 {
        //         assert_eq!(
        //             jolt_polynomials.bytecode.v_read_write[j].get_coeff(i),
        //             eval.bytecode.v_read_write[j]
        //         );
        //     }
        // }

        // jolt_oracle.reset_oracle();

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
            &preprocessing.generators,
            &preprocessing.bytecode,
            &jolt_polynomials.bytecode,
            &jolt_polynomials,
            &mut opening_accumulator,
            &mut transcript,
        );

        let instruction_proof = InstructionLookupsProof::prove(
            &preprocessing.generators,
            &mut jolt_polynomials,
            &preprocessing.instruction_lookups,
            &mut opening_accumulator,
            &mut transcript,
        );

        let memory_proof = ReadWriteMemoryProof::prove(
            &preprocessing.generators,
            &preprocessing.read_write_memory,
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

        // UniformSpartanProof::<
        //     C,
        //     <Self::Constraints as R1CSConstraints<C, F>>::Inputs,
        //     F,
        //     ProofTranscript,
        // >::streaming_prove::<PCS, Self::InstructionSet>(
        //     &r1cs_builder,
        //     &spartan_key,
        //     jolt_oracle,
        //     &mut opening_accumulator,
        //     &mut transcript,
        // );

        // Batch-prove all openings
        let opening_proof =
            opening_accumulator.reduce_and_prove::<PCS>(&preprocessing.generators, &mut transcript);

        drop_in_background_thread(jolt_polynomials);

        let jolt_proof = JoltProof {
            trace_length,
            program_io,
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
        (jolt_proof, jolt_commitments, debug_info)
    }

    #[tracing::instrument(skip_all)]
    fn verify(
        mut preprocessing: JoltPreprocessing<C, F, PCS, ProofTranscript>,
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
            &proof.program_io,
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
            proof.program_io,
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
