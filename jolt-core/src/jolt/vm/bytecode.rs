use ark_ff::Zero;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::rngs::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[cfg(test)]
use std::collections::HashSet;
use tracer::RV32IM;

use crate::field::JoltField;
use crate::jolt::instruction::JoltInstructionSet;
use crate::lasso::memory_checking::{
    Initializable, NoExogenousOpenings, StructuredPolynomialData, VerifierComputedOpening,
};
use crate::poly::commitment::commitment_scheme::{BatchType, CommitShape, CommitmentScheme};
use crate::poly::eq_poly::EqPolynomial;
use common::constants::{BYTES_PER_INSTRUCTION, RAM_START_ADDRESS};
use common::rv_trace::ELFInstruction;

use rayon::prelude::*;

use super::{JoltPolynomials, JoltTraceStep};
use crate::utils::transcript::Transcript;

use crate::{
    lasso::memory_checking::{MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier},
    poly::{dense_mlpoly::DensePolynomial, identity_poly::IdentityPolynomial},
};

#[derive(Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct BytecodeStuff<T: CanonicalSerialize + CanonicalDeserialize> {
    /// Read/write addresses for offline memory-checking.
    /// For offline memory-checking, each read is paired with a "virtual" write,
    /// so the read addresses and write addresses are the same.
    pub(crate) a_read_write: T,
    /// Read/write values for offline memory-checking.
    /// For offline memory-checking, each read is paired with a "virtual" write,
    /// so the read values and write values are the same. There are six values
    /// (address, bitflags, rd, rs1, rs2, imm) associated with each memory address.
    pub(crate) v_read_write: [T; 6],
    /// Read timestamps for offline memory-checking
    pub(crate) t_read: T,
    /// Final timestamps for offline memory-checking
    pub(crate) t_final: T,
    a_init_final: VerifierComputedOpening<T>,
    v_init_final: VerifierComputedOpening<[T; 6]>,
}

/// Note –– F: JoltField bound is not enforced.
///
/// See issue #112792 <https://github.com/rust-lang/rust/issues/112792>.
/// Adding #![feature(lazy_type_alias)] to the crate attributes seem to break
/// `alloy_sol_types`.
pub type BytecodePolynomials<F: JoltField> = BytecodeStuff<DensePolynomial<F>>;
/// Note –– F: JoltField bound is not enforced.
///
/// See issue #112792 <https://github.com/rust-lang/rust/issues/112792>.
/// Adding #![feature(lazy_type_alias)] to the crate attributes seem to break
/// `alloy_sol_types`.
pub type BytecodeOpenings<F: JoltField> = BytecodeStuff<F>;
/// Note –– PCS: CommitmentScheme bound is not enforced.
/// See issue #112792 <https://github.com/rust-lang/rust/issues/112792>.
/// Adding #![feature(lazy_type_alias)] to the crate attributes seem to break
pub type BytecodeCommitments<PCS: CommitmentScheme<ProofTranscript>, ProofTranscript: Transcript> =
    BytecodeStuff<PCS::Commitment>;

impl<F: JoltField, T: CanonicalSerialize + CanonicalDeserialize + Default>
    Initializable<T, BytecodePreprocessing<F>> for BytecodeStuff<T>
{
}

impl<T: CanonicalSerialize + CanonicalDeserialize> StructuredPolynomialData<T>
    for BytecodeStuff<T>
{
    fn read_write_values(&self) -> Vec<&T> {
        let mut values = vec![&self.a_read_write];
        values.extend(self.v_read_write.iter());
        values.push(&self.t_read);
        values
    }

    fn init_final_values(&self) -> Vec<&T> {
        vec![&self.t_final]
    }

    fn read_write_values_mut(&mut self) -> Vec<&mut T> {
        let mut values = vec![&mut self.a_read_write];
        values.extend(self.v_read_write.iter_mut());
        values.push(&mut self.t_read);
        values
    }

    fn init_final_values_mut(&mut self) -> Vec<&mut T> {
        vec![&mut self.t_final]
    }
}

pub type BytecodeProof<F, PCS, ProofTranscript> =
    MemoryCheckingProof<F, PCS, BytecodeOpenings<F>, NoExogenousOpenings, ProofTranscript>;

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
    imm: i64,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    virtual_sequence_remaining: Option<usize>,
}

impl BytecodeRow {
    pub fn new(address: usize, bitflags: u64, rd: u64, rs1: u64, rs2: u64, imm: i64) -> Self {
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
        // The load, store, and branch instructions need to do
        // field arithmetic with `imm` in constraints.rs,
        // whereas all other instructions operate on the raw bits
        // of `imm` (via lookup queries).
        let imm = match instruction.opcode {
            RV32IM::LW
            | RV32IM::SW
            | RV32IM::BEQ
            | RV32IM::BNE
            | RV32IM::BLT
            | RV32IM::BGE
            | RV32IM::BLTU
            | RV32IM::BGEU => instruction.imm.unwrap_or(0),
            _ => instruction.imm.unwrap_or(0) & u32::MAX as i64,
        };

        Self {
            address: instruction.address as usize,
            bitflags: Self::bitflags::<InstructionSet>(instruction),
            rd: instruction.rd.unwrap_or(0),
            rs1: instruction.rs1.unwrap_or(0),
            rs2: instruction.rs2.unwrap_or(0),
            imm,
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

#[derive(Clone)]
pub struct BytecodePreprocessing<F: JoltField> {
    /// Size of the (padded) bytecode.
    code_size: usize,
    /// MLE of init/final values. Bytecode is read-only data, so the final memory values are unchanged from
    /// the initial memory values. There are six values (address, bitflags, rd, rs1, rs2, imm)
    /// associated with each memory address, so `v_init_final` comprises six polynomials.
    v_init_final: [DensePolynomial<F>; 6],
    /// Maps the memory address of each instruction in the bytecode to its "virtual" address.
    /// See Section 6.1 of the Jolt paper, "Reflecting the program counter". The virtual address
    /// is the one used to keep track of the next (potentially virtual) instruction to execute.
    /// Key: (ELF address, virtual sequence index or 0)
    virtual_address_map: BTreeMap<(usize, usize), usize>,
}

impl<F: JoltField> BytecodePreprocessing<F> {
    #[tracing::instrument(skip_all, name = "BytecodePreprocessing::preprocess")]
    pub fn preprocess(mut bytecode: Vec<BytecodeRow>) -> Self {
        let mut virtual_address_map = BTreeMap::new();
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
            imm.push(F::from_i64(instruction.imm));
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

impl<F, PCS, ProofTranscript> BytecodeProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    #[tracing::instrument(skip_all, name = "BytecodePolynomials::new")]
    pub fn generate_witness<InstructionSet: JoltInstructionSet>(
        preprocessing: &BytecodePreprocessing<F>,
        trace: &mut Vec<JoltTraceStep<InstructionSet>>,
    ) -> BytecodePolynomials<F> {
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
            imm.push(F::from_i64(step.bytecode_row.imm));
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
        let mut init_tuples: HashSet<(u64, [F; 6], u64)> = HashSet::new();
        #[cfg(test)]
        let mut final_tuples: HashSet<(u64, [F; 6], u64)> = HashSet::new();

        #[cfg(test)]
        for (a, t) in t_final.Z.iter().enumerate() {
            init_tuples.insert((
                a as u64,
                [
                    preprocessing.v_init_final[0][a],
                    preprocessing.v_init_final[1][a],
                    preprocessing.v_init_final[2][a],
                    preprocessing.v_init_final[3][a],
                    preprocessing.v_init_final[4][a],
                    preprocessing.v_init_final[5][a],
                ],
                0,
            ));
            final_tuples.insert((
                a as u64,
                [
                    preprocessing.v_init_final[0][a],
                    preprocessing.v_init_final[1][a],
                    preprocessing.v_init_final[2][a],
                    preprocessing.v_init_final[3][a],
                    preprocessing.v_init_final[4][a],
                    preprocessing.v_init_final[5][a],
                ],
                t.to_u64().unwrap(),
            ));
        }

        #[cfg(test)]
        let mut read_tuples: HashSet<(u64, [F; 6], u64)> = HashSet::new();
        #[cfg(test)]
        let mut write_tuples: HashSet<(u64, [F; 6], u64)> = HashSet::new();

        #[cfg(test)]
        for (i, a) in a_read_write_usize.iter().enumerate() {
            read_tuples.insert((
                *a as u64,
                [
                    v_read_write[0][i],
                    v_read_write[1][i],
                    v_read_write[2][i],
                    v_read_write[3][i],
                    v_read_write[4][i],
                    v_read_write[5][i],
                ],
                t_read[i].to_u64().unwrap(),
            ));
            write_tuples.insert((
                *a as u64,
                [
                    v_read_write[0][i],
                    v_read_write[1][i],
                    v_read_write[2][i],
                    v_read_write[3][i],
                    v_read_write[4][i],
                    v_read_write[5][i],
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

        BytecodeStuff {
            a_read_write,
            v_read_write,
            t_read,
            t_final,
            a_init_final: None,
            v_init_final: None,
        }
    }

    #[tracing::instrument(skip_all, name = "BytecodePolynomials::validate_bytecode")]
    pub fn validate_bytecode(bytecode: &[BytecodeRow], trace: &[BytecodeRow]) {
        let mut bytecode_map: BTreeMap<usize, &BytecodeRow> = BTreeMap::new();

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

        let read_write_shape = CommitShape::new(max_trace_length, BatchType::Big);
        let init_final_shape = CommitShape::new(max_bytecode_size, BatchType::Small);

        vec![read_write_shape, init_final_shape]
    }
}

impl<F, PCS, ProofTranscript> MemoryCheckingProver<F, PCS, ProofTranscript>
    for BytecodeProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    type Polynomials = BytecodePolynomials<F>;
    type Openings = BytecodeOpenings<F>;
    type Commitments = BytecodeCommitments<PCS, ProofTranscript>;
    type Preprocessing = BytecodePreprocessing<F>;

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
        polynomials: &Self::Polynomials,
        _: &JoltPolynomials<F>,
        gamma: &F,
        tau: &F,
    ) -> ((Vec<F>, usize), (Vec<F>, usize)) {
        let num_ops = polynomials.a_read_write.len();
        let bytecode_size = preprocessing.v_init_final[0].len();

        let read_leaves: Vec<F> = (0..num_ops)
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

        let init_leaves: Vec<F> = (0..bytecode_size)
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

        // TODO(moodlezoup): avoid concat
        (
            ([read_leaves, write_leaves].concat(), 2),
            ([init_leaves, final_leaves].concat(), 2),
        )
    }

    fn protocol_name() -> &'static [u8] {
        b"Bytecode memory checking"
    }
}

impl<F, PCS, ProofTranscript> MemoryCheckingVerifier<F, PCS, ProofTranscript>
    for BytecodeProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    fn compute_verifier_openings(
        openings: &mut BytecodeOpenings<F>,
        preprocessing: &Self::Preprocessing,
        _r_read_write: &[F],
        r_init_final: &[F],
    ) {
        openings.a_init_final =
            Some(IdentityPolynomial::new(r_init_final.len()).evaluate(r_init_final));

        let chis = EqPolynomial::evals(r_init_final);
        openings.v_init_final = Some(
            preprocessing
                .v_init_final
                .par_iter()
                .map(|poly| poly.evaluate_at_chi(&chis))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        );
    }

    fn read_tuples(
        _: &BytecodePreprocessing<F>,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        vec![[
            openings.a_read_write,
            openings.v_read_write[0], // address
            openings.v_read_write[1], // opcode
            openings.v_read_write[2], // rd
            openings.v_read_write[3], // rs1
            openings.v_read_write[4], // rs2
            openings.v_read_write[5], // imm
            openings.t_read,
        ]]
    }
    fn write_tuples(
        _: &BytecodePreprocessing<F>,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
    ) -> Vec<Self::MemoryTuple> {
        vec![[
            openings.a_read_write,
            openings.v_read_write[0], // address
            openings.v_read_write[1], // opcode
            openings.v_read_write[2], // rd
            openings.v_read_write[3], // rs1
            openings.v_read_write[4], // rs2
            openings.v_read_write[5], // imm
            openings.t_read + F::one(),
        ]]
    }
    fn init_tuples(
        _: &BytecodePreprocessing<F>,
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
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
        openings: &Self::Openings,
        _: &NoExogenousOpenings,
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

#[cfg(test)]
mod tests {
    use crate::{jolt::vm::rv32i_vm::RV32I, poly::commitment::hyrax::HyraxScheme};

    use super::*;
    use crate::utils::transcript::KeccakTranscript;
    use ark_bn254::{Fr, G1Projective};
    use common::{
        constants::MEMORY_OPS_PER_INSTRUCTION,
        rv_trace::{MemoryOp, NUM_CIRCUIT_FLAGS},
    };
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
            circuit_flags: [false; NUM_CIRCUIT_FLAGS],
        }
    }

    fn to_ram_address(index: usize) -> usize {
        index * BYTES_PER_INSTRUCTION + RAM_START_ADDRESS as usize
    }

    #[test]
    fn bytecode_stuff_ordering() {
        let program = vec![
            BytecodeRow::new(to_ram_address(0), 2u64, 2u64, 2u64, 2u64, 2i64),
            BytecodeRow::new(to_ram_address(1), 4u64, 4u64, 4u64, 4u64, 4i64),
            BytecodeRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8i64),
            BytecodeRow::new(to_ram_address(3), 16u64, 16u64, 16u64, 16u64, 16i64),
        ];
        let preprocessing = BytecodePreprocessing::<Fr>::preprocess(program);
        BytecodeOpenings::<Fr>::test_ordering_consistency(&preprocessing);
    }

    #[test]
    #[should_panic]
    fn bytecode_validation_fake_trace() {
        let program = vec![
            BytecodeRow::new(to_ram_address(0), 2u64, 2u64, 2u64, 2u64, 2i64),
            BytecodeRow::new(to_ram_address(1), 4u64, 4u64, 4u64, 4u64, 4i64),
            BytecodeRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8i64),
            BytecodeRow::new(to_ram_address(3), 16u64, 16u64, 16u64, 16u64, 16i64),
            BytecodeRow::new(to_ram_address(4), 32u64, 32u64, 32u64, 32u64, 32i64),
        ];
        let trace = vec![
            BytecodeRow::new(to_ram_address(3), 16u64, 16u64, 16u64, 16u64, 16i64),
            BytecodeRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8i64),
            BytecodeRow::new(to_ram_address(5), 0u64, 0u64, 0u64, 0u64, 0i64), // no_op: shouldn't exist in pgoram
        ];
        BytecodeProof::<Fr, HyraxScheme<G1Projective, KeccakTranscript>, KeccakTranscript>::validate_bytecode(
            &program, &trace,
        );
    }

    #[test]
    #[should_panic]
    fn bytecode_validation_bad_prog_increment() {
        let program = vec![
            BytecodeRow::new(to_ram_address(0), 2u64, 2u64, 2u64, 2u64, 2i64),
            BytecodeRow::new(to_ram_address(1), 4u64, 4u64, 4u64, 4u64, 4i64),
            BytecodeRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8i64),
            BytecodeRow::new(to_ram_address(4), 16u64, 16u64, 16u64, 16u64, 16i64), // Increment by 2
        ];
        let trace = vec![
            BytecodeRow::new(to_ram_address(3), 16u64, 16u64, 16u64, 16u64, 16i64),
            BytecodeRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8i64),
        ];
        BytecodeProof::<Fr, HyraxScheme<G1Projective, KeccakTranscript>, KeccakTranscript>::validate_bytecode(
            &program, &trace,
        );
    }
}
