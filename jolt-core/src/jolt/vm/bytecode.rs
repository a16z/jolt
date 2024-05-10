use crate::poly::field::JoltField;
use rand::rngs::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, marker::PhantomData};

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
    bitflags: u64,
    /// Index of the destination register for this instruction (0 if register is unused).
    rd: u64,
    /// Index of the first source register for this instruction (0 if register is unused).
    rs1: u64,
    /// Index of the second source register for this instruction (0 if register is unused).
    rs2: u64,
    /// "Immediate" value for this instruction (0 if unused).
    imm: u64,
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
        }
    }
}

pub fn random_bytecode_trace(
    bytecode: &Vec<BytecodeRow>,
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
    /// so the read values and write values are the same. There are five values (bitflags, rd, rs1, rs2, imm)
    /// associated with each memory address, so `v_read_write` comprises five polynomials.
    pub(super) v_read_write: [DensePolynomial<F>; 5],
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
    /// the initial memory values. There are five values (bitflags, rd, rs1, rs2, imm)
    /// associated with each memory address, so `v_init_final` comprises five polynomials.
    v_init_final: [DensePolynomial<F>; 5],
}

impl<F: JoltField> BytecodePreprocessing<F> {
    #[tracing::instrument(skip_all, name = "BytecodePreprocessing::preprocess")]
    pub fn preprocess(mut bytecode: Vec<BytecodeRow>) -> Self {
        for instruction in bytecode.iter_mut() {
            assert!(instruction.address >= RAM_START_ADDRESS as usize);
            assert!(instruction.address % BYTES_PER_INSTRUCTION == 0);
            instruction.address -= RAM_START_ADDRESS as usize;
            instruction.address /= BYTES_PER_INSTRUCTION;

            // Account for no-op instruction prepended to bytecode
            instruction.address += 1;
        }

        // Bytecode: Prepend a single no-op instruction
        bytecode.insert(0, BytecodeRow::no_op(0));

        // Bytecode: Pad to nearest power of 2
        bytecode.resize(bytecode.len().next_power_of_two(), BytecodeRow::no_op(0));

        let max_bytecode_address = bytecode.iter().map(|instr| instr.address).max().unwrap();
        // Bytecode addresses are 0-indexed, so we add one to `max_bytecode_address`
        let code_size = (max_bytecode_address + 1).next_power_of_two();

        let v_init_final = to_v_polys(&bytecode);

        Self {
            v_init_final,
            code_size,
        }
    }
}

fn to_v_polys<F: JoltField>(rows: &Vec<BytecodeRow>) -> [DensePolynomial<F>; 5] {
    let len = rows.len().next_power_of_two();
    let mut bitflags = Vec::with_capacity(len);
    let mut rd = Vec::with_capacity(len);
    let mut rs1 = Vec::with_capacity(len);
    let mut rs2 = Vec::with_capacity(len);
    let mut imm = Vec::with_capacity(len);

    for row in rows {
        bitflags.push(F::from_u64(row.bitflags).unwrap());
        rd.push(F::from_u64(row.rd).unwrap());
        rs1.push(F::from_u64(row.rs1).unwrap());
        rs2.push(F::from_u64(row.rs2).unwrap());
        imm.push(F::from_u64(row.imm).unwrap());
    }
    // Padding
    bitflags.resize(len, F::zero());
    rd.resize(len, F::zero());
    rs1.resize(len, F::zero());
    rs2.resize(len, F::zero());
    imm.resize(len, F::zero());

    [
        DensePolynomial::new(bitflags),
        DensePolynomial::new(rd),
        DensePolynomial::new(rs1),
        DensePolynomial::new(rs2),
        DensePolynomial::new(imm),
    ]
}

impl<F: JoltField, C: CommitmentScheme<Field = F>> BytecodePolynomials<F, C> {
    #[tracing::instrument(skip_all, name = "BytecodePolynomials::new")]
    pub fn new(preprocessing: &BytecodePreprocessing<F>, mut trace: Vec<BytecodeRow>) -> Self {
        // Remap trace addresses
        for instruction in trace.iter_mut() {
            assert!(instruction.address >= RAM_START_ADDRESS as usize);
            assert!(instruction.address % BYTES_PER_INSTRUCTION == 0);
            instruction.address -= RAM_START_ADDRESS as usize;
            instruction.address /= BYTES_PER_INSTRUCTION;

            // Account for no-op instruction prepended to bytecode
            instruction.address += 1;
        }

        // Pad trace to nearest power of 2
        trace.resize(trace.len().next_power_of_two(), BytecodeRow::no_op(0));

        let num_ops = trace.len();

        let mut a_read_write_usize: Vec<usize> = vec![0; num_ops];
        let mut read_cts: Vec<usize> = vec![0; num_ops];
        let mut final_cts: Vec<usize> = vec![0; preprocessing.code_size];

        for (trace_index, trace) in trace.iter().enumerate() {
            let address = trace.address;
            debug_assert!(address < preprocessing.code_size);
            a_read_write_usize[trace_index] = address;
            let counter = final_cts[address];
            read_cts[trace_index] = counter;
            final_cts[address] = counter + 1;
        }

        let a_read_write = DensePolynomial::from_usize(&a_read_write_usize);
        let v_read_write = to_v_polys(&trace);
        let t_read = DensePolynomial::from_usize(&read_cts);
        let t_final = DensePolynomial::from_usize(&final_cts);

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

        // a_read_write, t_read, v_read_write (opcode, rs1, rs2, rd, imm)
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
    fn append_to_transcript(&self, label: &'static [u8], transcript: &mut ProofTranscript) {
        transcript.append_protocol_name(label);

        for commitment in &self.trace_commitments {
            commitment.append_to_transcript(b"trace", transcript);
        }

        self.t_final_commitment
            .append_to_transcript(b"final", transcript);
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

    // [a, opcode, rd, rs1, rs2, imm, t]
    type MemoryTuple = [F; 7];

    fn fingerprint(inputs: &Self::MemoryTuple, gamma: &F, tau: &F) -> F {
        let mut result = F::zero();
        let mut gamma_term = F::one();
        for input in inputs {
            result += *input * gamma_term;
            gamma_term *= gamma;
        }
        result - tau
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
            openings.v_read_write_openings[0], // opcode
            openings.v_read_write_openings[1], // rd
            openings.v_read_write_openings[2], // rs1
            openings.v_read_write_openings[3], // rs2
            openings.v_read_write_openings[4], // imm
            openings.t_read_opening,
        ]]
    }
    fn write_tuples(
        _: &BytecodePreprocessing<F>,
        openings: &Self::ReadWriteOpenings,
    ) -> Vec<Self::MemoryTuple> {
        vec![[
            openings.a_read_write_opening,
            openings.v_read_write_openings[0], // opcode
            openings.v_read_write_openings[1], // rd
            openings.v_read_write_openings[2], // rs1
            openings.v_read_write_openings[3], // rs2
            openings.v_read_write_openings[4], // imm
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
            v_init_final[0], // opcode
            v_init_final[1], // rd
            v_init_final[2], // rs1
            v_init_final[3], // rs2
            v_init_final[4], // imm
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
            v_init_final[0], // opcode
            v_init_final[1], // rd
            v_init_final[2], // rs1
            v_init_final[3], // rs2
            v_init_final[4], // imm
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
    v_read_write_openings: [F; 5],
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
        polynomials: &BytecodePolynomials<F, C>,
        opening_point: &[F],
        openings: &Self,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let mut combined_openings: Vec<F> =
            vec![openings.a_read_write_opening, openings.t_read_opening];
        combined_openings.extend(openings.v_read_write_openings.iter());

        C::batch_prove(
            &[
                &polynomials.a_read_write,
                &polynomials.t_read,
                &polynomials.v_read_write[0],
                &polynomials.v_read_write[1],
                &polynomials.v_read_write[2],
                &polynomials.v_read_write[3],
                &polynomials.v_read_write[4],
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
    v_init_final: Option<[F; 5]>,
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
        polynomials: &BytecodePolynomials<F, C>,
        opening_point: &[F],
        _openings: &Self,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        C::prove(&polynomials.t_final, opening_point, transcript)
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
    use crate::poly::commitment::hyrax::HyraxScheme;

    use super::*;
    use ark_bn254::{Fr, G1Projective};
    use std::collections::HashSet;

    fn get_difference<T: Clone + Eq + std::hash::Hash>(vec1: &[T], vec2: &[T]) -> Vec<T> {
        let set1: HashSet<_> = vec1.iter().cloned().collect();
        let set2: HashSet<_> = vec2.iter().cloned().collect();
        set1.symmetric_difference(&set2).cloned().collect()
    }

    #[test]
    fn bytecode_poly_leaf_construction() {
        let program = vec![
            BytecodeRow::new(to_ram_address(0), 2u64, 2u64, 2u64, 2u64, 2u64),
            BytecodeRow::new(to_ram_address(1), 4u64, 4u64, 4u64, 4u64, 4u64),
            BytecodeRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8u64),
            BytecodeRow::new(to_ram_address(3), 16u64, 16u64, 16u64, 16u64, 16u64),
        ];
        let trace = vec![
            BytecodeRow::new(to_ram_address(3), 16u64, 16u64, 16u64, 16u64, 16u64),
            BytecodeRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8u64),
        ];

        let preprocessing = BytecodePreprocessing::preprocess(program.clone());
        let polys: BytecodePolynomials<Fr, HyraxScheme<G1Projective>> =
            BytecodePolynomials::new(&preprocessing, trace);

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
        let trace = vec![
            BytecodeRow::new(to_ram_address(3), 16u64, 16u64, 16u64, 16u64, 16u64),
            BytecodeRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8u64),
        ];
        let commitment_shapes = BytecodePolynomials::<Fr, HyraxScheme<G1Projective>>::commit_shapes(
            program.len(),
            trace.len(),
        );

        let preprocessing = BytecodePreprocessing::preprocess(program.clone());
        let polys: BytecodePolynomials<Fr, HyraxScheme<G1Projective>> =
            BytecodePolynomials::new(&preprocessing, trace);

        let mut transcript = ProofTranscript::new(b"test_transcript");

        let generators = HyraxScheme::<G1Projective>::setup(&commitment_shapes);
        let commitments = polys.commit(&generators);
        let proof = BytecodeProof::prove_memory_checking(&preprocessing, &polys, &mut transcript);

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
        let trace = vec![
            BytecodeRow::new(to_ram_address(3), 16u64, 16u64, 16u64, 16u64, 16u64),
            BytecodeRow::new(to_ram_address(2), 8u64, 8u64, 8u64, 8u64, 8u64),
            BytecodeRow::new(to_ram_address(4), 32u64, 32u64, 32u64, 32u64, 32u64),
        ];

        let commit_shapes = BytecodePolynomials::<Fr, HyraxScheme<G1Projective>>::commit_shapes(
            program.len(),
            trace.len(),
        );
        let preprocessing = BytecodePreprocessing::preprocess(program.clone());
        let polys: BytecodePolynomials<Fr, HyraxScheme<G1Projective>> =
            BytecodePolynomials::new(&preprocessing, trace);
        let generators = HyraxScheme::<G1Projective>::setup(&commit_shapes);
        let commitments = polys.commit(&generators);

        let mut transcript = ProofTranscript::new(b"test_transcript");

        let proof = BytecodeProof::prove_memory_checking(&preprocessing, &polys, &mut transcript);

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
