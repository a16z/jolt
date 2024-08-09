use ark_ff::Zero;
use rand::rngs::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
#[cfg(test)]
use std::collections::HashSet;
use std::{collections::HashMap, marker::PhantomData};

use crate::field::JoltField;
use crate::jolt::instruction::JoltInstructionSet;
use crate::poly::commitment::commitment_scheme::{BatchType, CommitShape, CommitmentScheme};
use crate::poly::eq_poly::EqPolynomial;
use crate::utils::transcript::{AppendToTranscript, ProofTranscript};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::{BYTES_PER_INSTRUCTION, RAM_START_ADDRESS, REGISTER_COUNT};
use common::rv_trace::ELFInstruction;
use common::to_ram_address;

use rayon::prelude::*;

use crate::{
    lasso::memory_checking::{MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier},
    poly::{
        dense_mlpoly::DensePolynomial,
        identity_poly::IdentityPolynomial,
        structured_poly::{StructuredCommitment, StructuredOpeningProof},
    },
    utils::errors::ProofVerifyError,
};

use super::JoltTraceStep;

pub type BytecodeProof<F, C> = MemoryCheckingProof<
    F,
    C,
    BytecodePolynomials<F, C>,
    BytecodeReadWriteOpenings<F>,
    BytecodeInitFinalOpenings<F>,
>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BytecodeRow {
    /// Memory address as read from the ELF.
    address: usize,
    /// Packed instruction/circuit flags, used for r1cs
    pub bitflags: u64,
    /// Index of the destination register for this instruction (0 if register is unused).
    rd: u64,
    /// Index of the first source register for this instruction (0 if register is unused).
    rs1: u64,
    /// Index of the second source register for this instruction (0 if register is unused).
    rs2: u64,
    /// "Immediate" value for this instruction (0 if unused).
    imm: u64,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    virtual_sequence_remaining: Option<usize>,
}

impl BytecodeRow {
    pub fn new(address: usize, bitflags: u64, rd: u64, rs1: u64, rs2: u64, imm: u64) -> Self {
        Self {
            address,
            bitflags,
            rd,
            rs1,
            rs2,
            imm,
            virtual_sequence_remaining: None,
        }
    }

    pub fn no_op(address: usize) -> Self {
        Self {
            address,
            bitflags: 0,
            rd: 0,
            rs1: 0,
            rs2: 0,
            imm: 0,
            virtual_sequence_remaining: None,
        }
    }

    pub fn random(index: usize, rng: &mut StdRng) -> Self {
        Self {
            address: to_ram_address(index),
            bitflags: rng.next_u32() as u64, // Roughly how many flags there are
            rd: rng.next_u64() % REGISTER_COUNT,
            rs1: rng.next_u64() % REGISTER_COUNT,
            rs2: rng.next_u64() % REGISTER_COUNT,
            imm: rng.next_u64() % (1 << 20), // U-format instructions have 20-bit imm values
            virtual_sequence_remaining: None,
        }
    }

    /// Packs the instruction's circuit flags and instruction flags into a single u64 bitvector.
    /// The layout is:
    ///     circuit flags || instruction flags
    /// where instruction flags is a one-hot bitvector corresponding to the instruction's
    /// index in the `InstructionSet` enum.
    pub fn bitflags<InstructionSet>(instruction: &ELFInstruction) -> u64
    where
        InstructionSet: JoltInstructionSet,
    {
        let mut bitvector = 0;
        for flag in instruction.to_circuit_flags() {
            bitvector |= flag as u64;
            bitvector <<= 1;
        }

        // instruction flag
        if let Ok(jolt_instruction) = InstructionSet::try_from(instruction) {
            let instruction_index = InstructionSet::enum_index(&jolt_instruction);
            bitvector <<= instruction_index;
            bitvector |= 1;
            bitvector <<= InstructionSet::COUNT - instruction_index - 1;
        } else {
            bitvector <<= InstructionSet::COUNT - 1;
        }

        bitvector
    }

    pub fn from_instruction<InstructionSet>(instruction: &ELFInstruction) -> Self
    where
        InstructionSet: JoltInstructionSet,
    {
        Self {
            address: instruction.address as usize,
            bitflags: Self::bitflags::<InstructionSet>(instruction),
            rd: instruction.rd.unwrap_or(0),
            rs1: instruction.rs1.unwrap_or(0),
            rs2: instruction.rs2.unwrap_or(0),
            imm: instruction.imm.unwrap_or(0) as u64, // imm is always cast to its 32-bit repr, signed or unsigned
            virtual_sequence_remaining: instruction.virtual_sequence_remaining,
        }
    }
}

pub fn random_bytecode_trace(
    bytecode: &[BytecodeRow],
    num_ops: usize,
    rng: &mut StdRng,
) -> Vec<BytecodeRow> {
    let mut trace: Vec<BytecodeRow> = Vec::with_capacity(num_ops);
    for _ in 0..num_ops {
        trace.push(bytecode[rng.next_u64() as usize % bytecode.len()].clone());
    }
    trace
}

pub struct BytecodePolynomials<F: JoltField, C: CommitmentScheme<Field = F>> {
    _group: PhantomData<C>,
    /// MLE of read/write addresses. For offline memory checking, each read is paired with a "virtual" write,
    /// so the read addresses and write addresses are the same.
    pub(super) a_read_write: DensePolynomial<F>,
    /// MLE of read/write values. For offline memory checking, each read is paired with a "virtual" write,
    /// so the read values and write values are the same. There are six values (address, bitflags, rd, rs1, rs2, imm)
    /// associated with each memory address, so `v_read_write` comprises five polynomials.
    pub(super) v_read_write: [DensePolynomial<F>; 6],
    /// MLE of the read timestamps.
    pub(super) t_read: DensePolynomial<F>,
    /// MLE of the final timestamps.
    pub(super) t_final: DensePolynomial<F>,
}

#[derive(Clone)]
pub struct BytecodePreprocessing<F: JoltField> {
    /// Size of the (padded) bytecode.
    code_size: usize,
    /// MLE of init/final values. Bytecode is read-only data, so the final memory values are unchanged from
    /// the initial memory values. There are six values (address, bitflags, rd, rs1, rs2, imm)
    /// associated with each memory address, so `v_init_final` comprises five polynomials.
    v_init_final: [DensePolynomial<F>; 6],
    /// Maps the memory address of each instruction in the bytecode to its "virtual" address.
    /// See Section 6.1 of the Jolt paper, "Reflecting the program counter". The virtual address
    /// is the one used to keep track of the next (potentially virtual) instruction to execute.
    /// Key: (ELF address, virtual sequence index or 0)
    virtual_address_map: HashMap<(usize, usize), usize>,
}

impl<F: JoltField> BytecodePreprocessing<F> {
    #[tracing::instrument(skip_all, name = "BytecodePreprocessing::preprocess")]
    pub fn preprocess(mut bytecode: Vec<BytecodeRow>) -> Self {
        let mut virtual_address_map = HashMap::new();
        let mut virtual_address = 1; // Account for no-op instruction prepended to bytecode
        for instruction in bytecode.iter_mut() {
            assert!(instruction.address >= RAM_START_ADDRESS as usize);
            assert!(instruction.address % BYTES_PER_INSTRUCTION == 0);
            // Compress instruction address for more efficient commitment:
            instruction.address =
                1 + (instruction.address - RAM_START_ADDRESS as usize) / BYTES_PER_INSTRUCTION;
            assert_eq!(
                virtual_address_map.insert(
                    (
                        instruction.address,
                        instruction.virtual_sequence_remaining.unwrap_or(0)
                    ),
                    virtual_address
                ),
                None
            );
            virtual_address += 1;
        }

        // Bytecode: Prepend a single no-op instruction
        bytecode.insert(0, BytecodeRow::no_op(0));
        assert_eq!(virtual_address_map.insert((0, 0), 0), None);

        // Bytecode: Pad to nearest power of 2
        let code_size = bytecode.len().next_power_of_two();
        bytecode.resize(code_size, BytecodeRow::no_op(0));

        let mut address = vec![];
        let mut bitflags = vec![];
        let mut rd = vec![];
        let mut rs1 = vec![];
        let mut rs2 = vec![];
        let mut imm = vec![];

        for instruction in bytecode {
            address.push(F::from_u64(instruction.address as u64).unwrap());
            bitflags.push(F::from_u64(instruction.bitflags).unwrap());
            rd.push(F::from_u64(instruction.rd).unwrap());
            rs1.push(F::from_u64(instruction.rs1).unwrap());
            rs2.push(F::from_u64(instruction.rs2).unwrap());
            imm.push(F::from_u64(instruction.imm).unwrap());
        }

        let v_init_final = [
            DensePolynomial::new(address),
            DensePolynomial::new(bitflags),
            DensePolynomial::new(rd),
            DensePolynomial::new(rs1),
            DensePolynomial::new(rs2),
            DensePolynomial::new(imm),
        ];

        Self {
            v_init_final,
            code_size,
            virtual_address_map,
        }
    }
}

impl<F: JoltField, C: CommitmentScheme<Field = F>> BytecodePolynomials<F, C> {
    #[tracing::instrument(skip_all, name = "BytecodePolynomials::new")]
    pub fn new<InstructionSet: JoltInstructionSet>(
        preprocessing: &BytecodePreprocessing<F>,
        trace: &mut Vec<JoltTraceStep<InstructionSet>>,
    ) -> Self {
        let num_ops = trace.len();

        let mut a_read_write_usize: Vec<usize> = vec![0; num_ops];
        let mut read_cts: Vec<usize> = vec![0; num_ops];
        let mut final_cts: Vec<usize> = vec![0; preprocessing.code_size];

        for (step_index, step) in trace.iter_mut().enumerate() {
            if !step.bytecode_row.address.is_zero() {
                assert!(step.bytecode_row.address >= RAM_START_ADDRESS as usize);
                assert!(step.bytecode_row.address % BYTES_PER_INSTRUCTION == 0);
                // Compress instruction address for more efficient commitment:
                step.bytecode_row.address = 1
                    + (step.bytecode_row.address - RAM_START_ADDRESS as usize)
                        / BYTES_PER_INSTRUCTION;
            }

            let virtual_address = preprocessing
                .virtual_address_map
                .get(&(
                    step.bytecode_row.address,
                    step.bytecode_row.virtual_sequence_remaining.unwrap_or(0),
                ))
                .unwrap();
            a_read_write_usize[step_index] = *virtual_address;
            let counter = final_cts[*virtual_address];
            read_cts[step_index] = counter;
            final_cts[*virtual_address] = counter + 1;
        }

        let a_read_write = DensePolynomial::from_usize(&a_read_write_usize);

        let mut address = vec![];
        let mut bitflags = vec![];
        let mut rd = vec![];
        let mut rs1 = vec![];
        let mut rs2 = vec![];
        let mut imm = vec![];

        for step in trace {
            address.push(F::from_u64(step.bytecode_row.address as u64).unwrap());
            bitflags.push(F::from_u64(step.bytecode_row.bitflags).unwrap());
            rd.push(F::from_u64(step.bytecode_row.rd).unwrap());
            rs1.push(F::from_u64(step.bytecode_row.rs1).unwrap());
            rs2.push(F::from_u64(step.bytecode_row.rs2).unwrap());
            imm.push(F::from_u64(step.bytecode_row.imm).unwrap());
        }

        let v_read_write = [
            DensePolynomial::new(address),
            DensePolynomial::new(bitflags),
            DensePolynomial::new(rd),
            DensePolynomial::new(rs1),
            DensePolynomial::new(rs2),
            DensePolynomial::new(imm),
        ];
        let t_read: DensePolynomial<F> = DensePolynomial::from_usize(&read_cts);
        let t_final: DensePolynomial<F> = DensePolynomial::from_usize(&final_cts);

        #[cfg(test)]
        let mut init_tuples: HashSet<(u64, [u64; 6], u64)> = HashSet::new();
        #[cfg(test)]
        let mut final_tuples: HashSet<(u64, [u64; 6], u64)> = HashSet::new();

        #[cfg(test)]
        for (a, t) in t_final.Z.iter().enumerate() {
            init_tuples.insert((
                a as u64,
                [
                    preprocessing.v_init_final[0][a].to_u64().unwrap(),
                    preprocessing.v_init_final[1][a].to_u64().unwrap(),
                    preprocessing.v_init_final[2][a].to_u64().unwrap(),
                    preprocessing.v_init_final[3][a].to_u64().unwrap(),
                    preprocessing.v_init_final[4][a].to_u64().unwrap(),
                    preprocessing.v_init_final[5][a].to_u64().unwrap(),
                ],
                0,
            ));
            final_tuples.insert((
                a as u64,
                [
                    preprocessing.v_init_final[0][a].to_u64().unwrap(),
                    preprocessing.v_init_final[1][a].to_u64().unwrap(),
                    preprocessing.v_init_final[2][a].to_u64().unwrap(),
                    preprocessing.v_init_final[3][a].to_u64().unwrap(),
                    preprocessing.v_init_final[4][a].to_u64().unwrap(),
                    preprocessing.v_init_final[5][a].to_u64().unwrap(),
                ],
                t.to_u64().unwrap(),
            ));
        }

        #[cfg(test)]
        let mut read_tuples: HashSet<(u64, [u64; 6], u64)> = HashSet::new();
        #[cfg(test)]
        let mut write_tuples: HashSet<(u64, [u64; 6], u64)> = HashSet::new();

        #[cfg(test)]
        for (i, a) in a_read_write_usize.iter().enumerate() {
            read_tuples.insert((
                *a as u64,
                [
                    v_read_write[0][i].to_u64().unwrap(),
                    v_read_write[1][i].to_u64().unwrap(),
                    v_read_write[2][i].to_u64().unwrap(),
                    v_read_write[3][i].to_u64().unwrap(),
                    v_read_write[4][i].to_u64().unwrap(),
                    v_read_write[5][i].to_u64().unwrap(),
                ],
                t_read[i].to_u64().unwrap(),
            ));
            write_tuples.insert((
                *a as u64,
                [
                    v_read_write[0][i].to_u64().unwrap(),
                    v_read_write[1][i].to_u64().unwrap(),
                    v_read_write[2][i].to_u64().unwrap(),
                    v_read_write[3][i].to_u64().unwrap(),
                    v_read_write[4][i].to_u64().unwrap(),
                    v_read_write[5][i].to_u64().unwrap(),
                ],
                t_read[i].to_u64().unwrap() + 1,
            ));
        }

        #[cfg(test)]
        {
            let init_write: HashSet<_> = init_tuples.union(&write_tuples).collect();
            let read_final: HashSet<_> = read_tuples.union(&final_tuples).collect();
            let set_difference: Vec<_> = init_write.symmetric_difference(&read_final).collect();
            assert_eq!(set_difference.len(), 0);
        }

        Self {
            _group: PhantomData,
            a_read_write,
            v_read_write,
            t_read,
            t_final,
        }
    }

    #[tracing::instrument(skip_all, name = "BytecodePolynomials::get_polys_r1cs")]
    pub fn get_polys_r1cs(&self) -> (Vec<F>, Vec<F>) {
        let (a_read_write, v_read_write) = rayon::join(
            || self.a_read_write.evals(),
            || DensePolynomial::flatten(&self.v_read_write),
        );

        (a_read_write, v_read_write)
    }

    #[tracing::instrument(skip_all, name = "BytecodePolynomials::validate_bytecode")]
    pub fn validate_bytecode(bytecode: &[BytecodeRow], trace: &[BytecodeRow]) {
        let mut bytecode_map: HashMap<usize, &BytecodeRow> = HashMap::new();

        for bytecode_row in bytecode.iter() {
            bytecode_map.insert(bytecode_row.address, bytecode_row);
        }

        for trace_row in trace {
            assert_eq!(
                **bytecode_map
                    .get(&trace_row.address)
                    .expect("couldn't find in bytecode"),
                *trace_row
            );
        }
    }

    /// Computes the shape of all commitment for use in PCS::setup().
    pub fn commit_shapes(max_bytecode_size: usize, max_trace_length: usize) -> Vec<CommitShape> {
        // Account for no-op prepended to bytecode
        let max_bytecode_size = (max_bytecode_size + 1).next_power_of_two();
        let max_trace_length = max_trace_length.next_power_of_two();

        // a_read_write, t_read, v_read_write (address, opcode, rs1, rs2, rd, imm)
        let read_write_gen_shape = CommitShape::new(max_trace_length, BatchType::Big);

        // t_final
        let init_final_gen_shape = CommitShape::new(max_bytecode_size, BatchType::Small);

        vec![read_write_gen_shape, init_final_gen_shape]
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct BytecodeCommitment<C: CommitmentScheme> {
    pub trace_commitments: Vec<C::Commitment>,
    pub t_final_commitment: C::Commitment,
}

impl<C: CommitmentScheme> AppendToTranscript for BytecodeCommitment<C> {
    fn append_to_transcript(&self, transcript: &mut ProofTranscript) {
        transcript.append_protocol_name(b"Bytecode Commitments");

        for commitment in &self.trace_commitments {
            commitment.append_to_transcript(transcript);
        }

        self.t_final_commitment.append_to_transcript(transcript);
    }
}

impl<F, C> StructuredCommitment<C> for BytecodePolynomials<F, C>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    type Commitment = BytecodeCommitment<C>;

    #[tracing::instrument(skip_all, name = "BytecodePolynomials::commit")]
    fn commit(&self, generators: &C::Setup) -> Self::Commitment {
        let trace_polys = vec![
            &self.a_read_write,
            &self.t_read, // t_read isn't used in r1cs, but it's cleaner to commit to it as a rectangular matrix alongside everything else
            &self.v_read_write[0],
            &self.v_read_write[1],
            &self.v_read_write[2],
            &self.v_read_write[3],
            &self.v_read_write[4],
            &self.v_read_write[5],
        ];
        let trace_commitments = C::batch_commit_polys_ref(&trace_polys, generators, BatchType::Big);

        let t_final_commitment = C::commit(&self.t_final, generators);

        Self::Commitment {
            trace_commitments,
            t_final_commitment,
        }
    }
}

impl<F, C> MemoryCheckingProver<F, C, BytecodePolynomials<F, C>> for BytecodeProof<F, C>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    type Preprocessing = BytecodePreprocessing<F>;
    type ReadWriteOpenings = BytecodeReadWriteOpenings<F>;
    type InitFinalOpenings = BytecodeInitFinalOpenings<F>;

    // [virtual_address, elf_address, opcode, rd, rs1, rs2, imm, t]
    type MemoryTuple = [F; 8];

    fn fingerprint(inputs: &Self::MemoryTuple, gamma: &F, tau: &F) -> F {
        let mut result = F::zero();
        let mut gamma_term = F::one();
        for input in inputs {
            result += *input * gamma_term;
            gamma_term *= *gamma;
        }
        result - *tau
    }

    #[tracing::instrument(skip_all, name = "BytecodePolynomials::compute_leaves")]
    fn compute_leaves(
        preprocessing: &BytecodePreprocessing<F>,
        polynomials: &BytecodePolynomials<F, C>,
        gamma: &F,
        tau: &F,
    ) -> (Vec<Vec<F>>, Vec<Vec<F>>) {
        let num_ops = polynomials.a_read_write.len();
        let bytecode_size = preprocessing.v_init_final[0].len();

        let read_leaves = (0..num_ops)
            .into_par_iter()
            .map(|i| {
                Self::fingerprint(
                    &[
                        polynomials.a_read_write[i],
                        polynomials.v_read_write[0][i],
                        polynomials.v_read_write[1][i],
                        polynomials.v_read_write[2][i],
                        polynomials.v_read_write[3][i],
                        polynomials.v_read_write[4][i],
                        polynomials.v_read_write[5][i],
                        polynomials.t_read[i],
                    ],
                    gamma,
                    tau,
                )
            })
            .collect();

        let init_leaves = (0..bytecode_size)
            .into_par_iter()
            .map(|i| {
                Self::fingerprint(
                    &[
                        F::from_u64(i as u64).unwrap(),
                        preprocessing.v_init_final[0][i],
                        preprocessing.v_init_final[1][i],
                        preprocessing.v_init_final[2][i],
                        preprocessing.v_init_final[3][i],
                        preprocessing.v_init_final[4][i],
                        preprocessing.v_init_final[5][i],
                        F::zero(),
                    ],
                    gamma,
                    tau,
                )
            })
            .collect();

        // TODO(moodlezoup): Compute write_leaves from read_leaves
        let write_leaves = (0..num_ops)
            .into_par_iter()
            .map(|i| {
                Self::fingerprint(
                    &[
                        polynomials.a_read_write[i],
                        polynomials.v_read_write[0][i],
                        polynomials.v_read_write[1][i],
                        polynomials.v_read_write[2][i],
                        polynomials.v_read_write[3][i],
                        polynomials.v_read_write[4][i],
                        polynomials.v_read_write[5][i],
                        polynomials.t_read[i] + F::one(),
                    ],
                    gamma,
                    tau,
                )
            })
            .collect();

        // TODO(moodlezoup): Compute final_leaves from init_leaves
        let final_leaves = (0..bytecode_size)
            .into_par_iter()
            .map(|i| {
                Self::fingerprint(
                    &[
                        F::from_u64(i as u64).unwrap(),
                        preprocessing.v_init_final[0][i],
                        preprocessing.v_init_final[1][i],
                        preprocessing.v_init_final[2][i],
                        preprocessing.v_init_final[3][i],
                        preprocessing.v_init_final[4][i],
                        preprocessing.v_init_final[5][i],
                        polynomials.t_final[i],
                    ],
                    gamma,
                    tau,
                )
            })
            .collect();

        (
            vec![read_leaves, write_leaves],
            vec![init_leaves, final_leaves],
        )
    }

    fn protocol_name() -> &'static [u8] {
        b"Bytecode memory checking"
    }
}

impl<F, C> MemoryCheckingVerifier<F, C, BytecodePolynomials<F, C>> for BytecodeProof<F, C>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    fn read_tuples(
        _: &BytecodePreprocessing<F>,
        openings: &Self::ReadWriteOpenings,
    ) -> Vec<Self::MemoryTuple> {
        vec![[
            openings.a_read_write_opening,
            openings.v_read_write_openings[0], // address
            openings.v_read_write_openings[1], // opcode
            openings.v_read_write_openings[2], // rd
            openings.v_read_write_openings[3], // rs1
            openings.v_read_write_openings[4], // rs2
            openings.v_read_write_openings[5], // imm
            openings.t_read_opening,
        ]]
    }
    fn write_tuples(
        _: &BytecodePreprocessing<F>,
        openings: &Self::ReadWriteOpenings,
    ) -> Vec<Self::MemoryTuple> {
        vec![[
            openings.a_read_write_opening,
            openings.v_read_write_openings[0], // address
            openings.v_read_write_openings[1], // opcode
            openings.v_read_write_openings[2], // rd
            openings.v_read_write_openings[3], // rs1
            openings.v_read_write_openings[4], // rs2
            openings.v_read_write_openings[5], // imm
            openings.t_read_opening + F::one(),
        ]]
    }
    fn init_tuples(
        _: &BytecodePreprocessing<F>,
        openings: &Self::InitFinalOpenings,
    ) -> Vec<Self::MemoryTuple> {
        let v_init_final = openings.v_init_final.unwrap();
        vec![[
            openings.a_init_final.unwrap(),
            v_init_final[0], // address
            v_init_final[1], // opcode
            v_init_final[2], // rd
            v_init_final[3], // rs1
            v_init_final[4], // rs2
            v_init_final[5], // imm
            F::zero(),
        ]]
    }
    fn final_tuples(
        _: &BytecodePreprocessing<F>,
        openings: &Self::InitFinalOpenings,
    ) -> Vec<Self::MemoryTuple> {
        let v_init_final = openings.v_init_final.unwrap();
        vec![[
            openings.a_init_final.unwrap(),
            v_init_final[0], // address
            v_init_final[1], // opcode
            v_init_final[2], // rd
            v_init_final[3], // rs1
            v_init_final[4], // rs2
            v_init_final[5], // imm
            openings.t_final,
        ]]
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct BytecodeReadWriteOpenings<F>
where
    F: JoltField,
{
    /// Evaluation of the a_read_write polynomial at the opening point.
    a_read_write_opening: F,
    /// Evaluation of the v_read_write polynomials at the opening point.
    v_read_write_openings: [F; 6],
    /// Evaluation of the t_read polynomial at the opening point.
    t_read_opening: F,
}

impl<F, C> StructuredOpeningProof<F, C, BytecodePolynomials<F, C>> for BytecodeReadWriteOpenings<F>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    type Proof = C::BatchedProof;

    #[tracing::instrument(skip_all, name = "BytecodeReadWriteOpenings::open")]
    fn open(polynomials: &BytecodePolynomials<F, C>, opening_point: &[F]) -> Self {
        let chis = EqPolynomial::evals(opening_point);
        Self {
            a_read_write_opening: polynomials.a_read_write.evaluate_at_chi(&chis),
            v_read_write_openings: polynomials
                .v_read_write
                .par_iter()
                .map(|poly| poly.evaluate_at_chi(&chis))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            t_read_opening: polynomials.t_read.evaluate_at_chi(&chis),
        }
    }

    #[tracing::instrument(skip_all, name = "BytecodeReadWriteOpenings::prove_openings")]
    fn prove_openings(
        generators: &C::Setup,
        polynomials: &BytecodePolynomials<F, C>,
        opening_point: &[F],
        openings: &Self,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let mut combined_openings: Vec<F> =
            vec![openings.a_read_write_opening, openings.t_read_opening];
        combined_openings.extend(openings.v_read_write_openings.iter());

        C::batch_prove(
            generators,
            &[
                &polynomials.a_read_write,
                &polynomials.t_read,
                &polynomials.v_read_write[0],
                &polynomials.v_read_write[1],
                &polynomials.v_read_write[2],
                &polynomials.v_read_write[3],
                &polynomials.v_read_write[4],
                &polynomials.v_read_write[5],
            ],
            opening_point,
            &combined_openings,
            BatchType::Big,
            transcript,
        )
    }

    fn verify_openings(
        &self,
        generators: &C::Setup,
        opening_proof: &Self::Proof,
        commitment: &BytecodeCommitment<C>,
        opening_point: &[F],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let mut combined_openings: Vec<F> = vec![self.a_read_write_opening, self.t_read_opening];
        combined_openings.extend(self.v_read_write_openings.iter());

        C::batch_verify(
            opening_proof,
            generators,
            opening_point,
            &combined_openings,
            &commitment.trace_commitments.iter().collect::<Vec<_>>(),
            transcript,
        )
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct BytecodeInitFinalOpenings<F>
where
    F: JoltField,
{
    /// Evaluation of the a_init_final polynomial at the opening point. Computed by the verifier in `compute_verifier_openings`.
    a_init_final: Option<F>,
    /// Evaluation of the v_init/final polynomials at the opening point. Computed by the verifier in `compute_verifier_openings`.
    v_init_final: Option<[F; 6]>,
    /// Evaluation of the t_final polynomial at the opening point.
    t_final: F,
}

impl<F, C> StructuredOpeningProof<F, C, BytecodePolynomials<F, C>> for BytecodeInitFinalOpenings<F>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    type Preprocessing = BytecodePreprocessing<F>;
    type Proof = C::Proof;

    #[tracing::instrument(skip_all, name = "BytecodeInitFinalOpenings::open")]
    fn open(polynomials: &BytecodePolynomials<F, C>, opening_point: &[F]) -> Self {
        Self {
            a_init_final: None,
            v_init_final: None,
            t_final: polynomials.t_final.evaluate(opening_point),
        }
    }

    #[tracing::instrument(skip_all, name = "BytecodeInitFinalOpenings::prove_openings")]
    fn prove_openings(
        generators: &C::Setup,
        polynomials: &BytecodePolynomials<F, C>,
        opening_point: &[F],
        _openings: &Self,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        C::prove(generators, &polynomials.t_final, opening_point, transcript)
    }

    fn compute_verifier_openings(
        &mut self,
        preprocessing: &BytecodePreprocessing<F>,
        opening_point: &[F],
    ) {
        self.a_init_final =
            Some(IdentityPolynomial::new(opening_point.len()).evaluate(opening_point));

        let chis = EqPolynomial::evals(opening_point);
        self.v_init_final = Some(
            preprocessing
                .v_init_final
                .par_iter()
                .map(|poly| poly.evaluate_at_chi(&chis))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        );
    }

    fn verify_openings(
        &self,
        generators: &C::Setup,
        opening_proof: &Self::Proof,
        commitment: &BytecodeCommitment<C>,
        opening_point: &[F],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        C::verify(
            opening_proof,
            generators,
            transcript,
            opening_point,
            &self.t_final,
            &commitment.t_final_commitment,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::{jolt::vm::rv32i_vm::RV32I, poly::commitment::hyrax::HyraxScheme};

    use super::*;
    use ark_bn254::{Fr, G1Projective};
    use common::{constants::MEMORY_OPS_PER_INSTRUCTION, rv_trace::MemoryOp};
    use std::collections::HashSet;

    fn get_difference<T: Clone + Eq + std::hash::Hash>(vec1: &[T], vec2: &[T]) -> Vec<T> {
        let set1: HashSet<_> = vec1.iter().cloned().collect();
        let set2: HashSet<_> = vec2.iter().cloned().collect();
        set1.symmetric_difference(&set2).cloned().collect()
    }

    fn trace_step(bytecode_row: BytecodeRow) -> JoltTraceStep<RV32I> {
        JoltTraceStep {
            instruction_lookup: None,
            memory_ops: [MemoryOp::noop_read(); MEMORY_OPS_PER_INSTRUCTION],
            bytecode_row,
        }
    }

    #[test]
    fn bytecode_poly_leaf_construction() {
        let program = vec![
            BytecodeRow::new(to_ram_address(0), 2u64, 2u64, 2u64, 2u64, 2u64),
            BytecodeRow::new(to_ram_address(1), 4u64, 4u64, 4u64, 4u64, 4u64),
            BytecodeRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8u64),
            BytecodeRow::new(to_ram_address(3), 16u64, 16u64, 16u64, 16u64, 16u64),
        ];
        let mut trace = vec![
            trace_step(BytecodeRow::new(
                to_ram_address(3),
                16u64,
                16u64,
                16u64,
                16u64,
                16u64,
            )),
            trace_step(BytecodeRow::new(
                to_ram_address(2),
                8u64,
                8u64,
                8u64,
                8u64,
                8u64,
            )),
        ];

        let preprocessing = BytecodePreprocessing::preprocess(program.clone());
        let polys: BytecodePolynomials<Fr, HyraxScheme<G1Projective>> =
            BytecodePolynomials::new::<RV32I>(&preprocessing, &mut trace);

        let (gamma, tau) = (&Fr::from(100), &Fr::from(35));
        let (read_write_leaves, init_final_leaves) =
            BytecodeProof::compute_leaves(&preprocessing, &polys, gamma, tau);
        let init_leaves = &init_final_leaves[0];
        let read_leaves = &read_write_leaves[0];
        let write_leaves = &read_write_leaves[1];
        let final_leaves = &init_final_leaves[1];

        let read_final_leaves = [read_leaves.clone(), final_leaves.clone()].concat();
        let init_write_leaves = [init_leaves.clone(), write_leaves.clone()].concat();
        let difference: Vec<Fr> = get_difference(&read_final_leaves, &init_write_leaves);
        assert_eq!(difference.len(), 0);
    }

    #[test]
    fn e2e_memchecking() {
        let program = vec![
            BytecodeRow::new(to_ram_address(0), 2u64, 2u64, 2u64, 2u64, 2u64),
            BytecodeRow::new(to_ram_address(1), 4u64, 4u64, 4u64, 4u64, 4u64),
            BytecodeRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8u64),
            BytecodeRow::new(to_ram_address(3), 16u64, 16u64, 16u64, 16u64, 16u64),
        ];
        let mut trace = vec![
            trace_step(BytecodeRow::new(
                to_ram_address(3),
                16u64,
                16u64,
                16u64,
                16u64,
                16u64,
            )),
            trace_step(BytecodeRow::new(
                to_ram_address(2),
                8u64,
                8u64,
                8u64,
                8u64,
                8u64,
            )),
        ];
        let commitment_shapes = BytecodePolynomials::<Fr, HyraxScheme<G1Projective>>::commit_shapes(
            program.len(),
            trace.len(),
        );

        let preprocessing = BytecodePreprocessing::preprocess(program.clone());
        let polys: BytecodePolynomials<Fr, HyraxScheme<G1Projective>> =
            BytecodePolynomials::new(&preprocessing, &mut trace);

        let mut transcript = ProofTranscript::new(b"test_transcript");

        let generators = HyraxScheme::<G1Projective>::setup(&commitment_shapes);
        let commitments = polys.commit(&generators);
        let proof = BytecodeProof::prove_memory_checking(
            &generators,
            &preprocessing,
            &polys,
            &mut transcript,
        );

        let mut transcript = ProofTranscript::new(b"test_transcript");
        BytecodeProof::verify_memory_checking(
            &preprocessing,
            &generators,
            proof,
            &commitments,
            &mut transcript,
        )
        .expect("proof should verify");
    }

    #[test]
    fn e2e_mem_checking_non_pow_2() {
        let program = vec![
            BytecodeRow::new(to_ram_address(0), 2u64, 2u64, 2u64, 2u64, 2u64),
            BytecodeRow::new(to_ram_address(1), 4u64, 4u64, 4u64, 4u64, 4u64),
            BytecodeRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8u64),
            BytecodeRow::new(to_ram_address(3), 16u64, 16u64, 16u64, 16u64, 16u64),
            BytecodeRow::new(to_ram_address(4), 32u64, 32u64, 32u64, 32u64, 32u64),
        ];
        let mut trace = vec![
            trace_step(BytecodeRow::new(
                to_ram_address(3),
                16u64,
                16u64,
                16u64,
                16u64,
                16u64,
            )),
            trace_step(BytecodeRow::new(
                to_ram_address(2),
                8u64,
                8u64,
                8u64,
                8u64,
                8u64,
            )),
            trace_step(BytecodeRow::new(
                to_ram_address(4),
                32u64,
                32u64,
                32u64,
                32u64,
                32u64,
            )),
        ];
        JoltTraceStep::pad(&mut trace);

        let commit_shapes = BytecodePolynomials::<Fr, HyraxScheme<G1Projective>>::commit_shapes(
            program.len(),
            trace.len(),
        );
        let preprocessing = BytecodePreprocessing::preprocess(program.clone());
        let polys: BytecodePolynomials<Fr, HyraxScheme<G1Projective>> =
            BytecodePolynomials::new(&preprocessing, &mut trace);
        let generators = HyraxScheme::<G1Projective>::setup(&commit_shapes);
        let commitments = polys.commit(&generators);

        let mut transcript = ProofTranscript::new(b"test_transcript");

        let proof = BytecodeProof::prove_memory_checking(
            &generators,
            &preprocessing,
            &polys,
            &mut transcript,
        );

        let mut transcript = ProofTranscript::new(b"test_transcript");
        BytecodeProof::verify_memory_checking(
            &preprocessing,
            &generators,
            proof,
            &commitments,
            &mut transcript,
        )
        .expect("should verify");
    }

    #[test]
    #[should_panic]
    fn bytecode_validation_fake_trace() {
        let program = vec![
            BytecodeRow::new(to_ram_address(0), 2u64, 2u64, 2u64, 2u64, 2u64),
            BytecodeRow::new(to_ram_address(1), 4u64, 4u64, 4u64, 4u64, 4u64),
            BytecodeRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8u64),
            BytecodeRow::new(to_ram_address(3), 16u64, 16u64, 16u64, 16u64, 16u64),
            BytecodeRow::new(to_ram_address(4), 32u64, 32u64, 32u64, 32u64, 32u64),
        ];
        let trace = vec![
            BytecodeRow::new(to_ram_address(3), 16u64, 16u64, 16u64, 16u64, 16u64),
            BytecodeRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8u64),
            BytecodeRow::new(to_ram_address(5), 0u64, 0u64, 0u64, 0u64, 0u64), // no_op: shouldn't exist in pgoram
        ];
        BytecodePolynomials::<Fr, HyraxScheme<G1Projective>>::validate_bytecode(&program, &trace);
    }

    #[test]
    #[should_panic]
    fn bytecode_validation_bad_prog_increment() {
        let program = vec![
            BytecodeRow::new(to_ram_address(0), 2u64, 2u64, 2u64, 2u64, 2u64),
            BytecodeRow::new(to_ram_address(1), 4u64, 4u64, 4u64, 4u64, 4u64),
            BytecodeRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8u64),
            BytecodeRow::new(to_ram_address(4), 16u64, 16u64, 16u64, 16u64, 16u64), // Increment by 2
        ];
        let trace = vec![
            BytecodeRow::new(to_ram_address(3), 16u64, 16u64, 16u64, 16u64, 16u64),
            BytecodeRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8u64),
        ];
        BytecodePolynomials::<Fr, HyraxScheme<G1Projective>>::validate_bytecode(&program, &trace);
    }
}
