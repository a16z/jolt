use ark_ff::Zero;
use rand::rngs::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[cfg(test)]
use std::collections::HashSet;

use crate::field::JoltField;
use crate::jolt::instruction::JoltInstructionSet;
use crate::lasso::memory_checking::{NoAdditionalWitness, StructuredPolynomialData};
use crate::poly::commitment::commitment_scheme::{BatchType, CommitShape, CommitmentScheme};
use crate::poly::eq_poly::EqPolynomial;
use common::constants::{BYTES_PER_INSTRUCTION, RAM_START_ADDRESS, REGISTER_COUNT};
use common::rv_trace::ELFInstruction;
use common::to_ram_address;

use rayon::prelude::*;

use crate::{
    lasso::memory_checking::{MemoryCheckingProof, MemoryCheckingProver, MemoryCheckingVerifier},
    poly::{dense_mlpoly::DensePolynomial, identity_poly::IdentityPolynomial},
    utils::errors::ProofVerifyError,
};

use super::JoltTraceStep;

pub struct BytecodeStuff<T> {
    a_read_write: T,
    pub(crate) v_read_write: [T; 6],
    t_read: T,
    t_final: T,

    a_init_final: Option<T>,
    v_init_final: Option<[T; 6]>,
}
pub type BytecodePolynomials<F: JoltField> = BytecodeStuff<DensePolynomial<F>>;
pub type BytecodeOpenings<F: JoltField> = BytecodeStuff<F>;
pub type BytecodeCommitments<PCS: CommitmentScheme> = BytecodeStuff<PCS::Commitment>;

impl<T> StructuredPolynomialData<T> for BytecodeStuff<T> {
    fn read_write_values(&self) -> Vec<&T> {
        vec![
            &self.a_read_write,
            &self.v_read_write[0],
            &self.v_read_write[1],
            &self.v_read_write[2],
            &self.v_read_write[3],
            &self.v_read_write[4],
            &self.v_read_write[5],
            &self.t_read,
        ]
    }

    fn init_final_values(&self) -> Vec<&T> {
        vec![&self.t_final]
    }

    fn read_write_values_mut(&mut self) -> Vec<&mut T> {
        vec![
            &mut self.a_read_write,
            &mut self.v_read_write[0],
            &mut self.v_read_write[1],
            &mut self.v_read_write[2],
            &mut self.v_read_write[3],
            &mut self.v_read_write[4],
            &mut self.v_read_write[5],
            &mut self.t_read,
        ]
    }

    fn init_final_values_mut(&mut self) -> Vec<&mut T> {
        vec![&mut self.t_final]
    }
}

pub type BytecodeProof<F, PCS> = MemoryCheckingProof<F, PCS, BytecodeOpenings<F>>;

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

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> BytecodeProof<F, PCS> {
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

        // a_read_write, t_read, v_read_write (address, opcode, rs1, rs2, rd, imm)
        let read_write_gen_shape = CommitShape::new(max_trace_length, BatchType::Big);

        // t_final
        let init_final_gen_shape = CommitShape::new(max_bytecode_size, BatchType::Small);

        vec![read_write_gen_shape, init_final_gen_shape]
    }
}

impl<F, PCS> MemoryCheckingProver<F, PCS> for BytecodeProof<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    type StructuredData<T> = BytecodeStuff<T> where T: Sync;
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
        _: &NoAdditionalWitness,
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

impl<F, PCS> MemoryCheckingVerifier<F, PCS> for BytecodeProof<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn compute_verifier_openings(
        proof: &mut MemoryCheckingProof<F, PCS, BytecodeOpenings<F>>,
        preprocessing: &Self::Preprocessing,
        _r_read_write: &[F],
        r_init_final: &[F],
    ) {
        proof.openings.a_init_final =
            Some(IdentityPolynomial::new(r_init_final.len()).evaluate(r_init_final));

        let chis = EqPolynomial::evals(r_init_final);
        proof.openings.v_init_final = Some(
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
        openings: &Self::Openings,
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
        openings: &Self::Openings,
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
    use crate::{
        jolt::vm::rv32i_vm::RV32I,
        poly::{commitment::hyrax::HyraxScheme, opening_proof::ProverOpeningAccumulator},
    };

    use super::*;
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
        let mut opening_accumulator = ProverOpeningAccumulator::new();
        let proof = BytecodeProof::prove_memory_checking(
            &generators,
            &preprocessing,
            &polys,
            &mut opening_accumulator,
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

        let mut opening_accumulator = ProverOpeningAccumulator::new();
        let proof = BytecodeProof::prove_memory_checking(
            &generators,
            &preprocessing,
            &polys,
            &mut opening_accumulator,
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
