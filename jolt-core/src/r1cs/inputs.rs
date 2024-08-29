#![allow(
    clippy::len_without_is_empty,
    clippy::type_complexity,
    clippy::too_many_arguments
)]

use crate::impl_r1cs_input_lc_conversions;
use crate::jolt::instruction::JoltInstructionSet;
use crate::jolt::vm::rv32i_vm::RV32I;
use crate::jolt::vm::{JoltPolynomials, JoltTraceStep};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::utils::transcript::ProofTranscript;

use super::key::UniformSpartanKey;
use super::spartan::{SpartanError, UniformSpartanProof};

use crate::field::JoltField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::log2;
use common::constants::RAM_OPS_PER_INSTRUCTION;
use common::rv_trace::{CircuitFlags, NUM_CIRCUIT_FLAGS};
use std::fmt::Debug;
use std::hash::Hash;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

pub struct AuxPolynomials<F: JoltField> {
    pub left_lookup_operand: DensePolynomial<F>,
    pub right_lookup_operand: DensePolynomial<F>,
    pub imm_signed: DensePolynomial<F>,
    pub product: DensePolynomial<F>,
    pub relevant_y_chunks: Vec<DensePolynomial<F>>,
    pub write_lookup_output_to_rd: DensePolynomial<F>,
    pub write_pc_to_rd: DensePolynomial<F>,
    pub next_pc_jump: DensePolynomial<F>,
    pub should_branch: DensePolynomial<F>,
    pub next_pc: DensePolynomial<F>,
}

pub struct R1CSPolynomials<F: JoltField> {
    pub chunks_x: Vec<DensePolynomial<F>>,
    pub chunks_y: Vec<DensePolynomial<F>>,
    pub circuit_flags: [DensePolynomial<F>; NUM_CIRCUIT_FLAGS],
    pub aux: Option<AuxPolynomials<F>>,
}

impl<F: JoltField> R1CSPolynomials<F> {
    pub fn new<
        const C: usize,
        const M: usize,
        InstructionSet: JoltInstructionSet,
        I: ConstraintInput,
    >(
        trace: &[JoltTraceStep<InstructionSet>],
    ) -> Self {
        let log_M = log2(M) as usize;

        let mut chunks_x = vec![unsafe_allocate_zero_vec(trace.len()); C];
        let mut chunks_y = vec![unsafe_allocate_zero_vec(trace.len()); C];
        let mut circuit_flags = vec![unsafe_allocate_zero_vec(trace.len()); NUM_CIRCUIT_FLAGS];

        // TODO(moodlezoup): Can be parallelized
        for (step_index, step) in trace.iter().enumerate() {
            if let Some(instr) = &step.instruction_lookup {
                let (x, y) = instr.operand_chunks(C, log_M);
                for i in 0..C {
                    chunks_x[i][step_index] = F::from_u64(x[i]).unwrap();
                    chunks_y[i][step_index] = F::from_u64(y[i]).unwrap();
                }
            }

            for j in 0..NUM_CIRCUIT_FLAGS {
                if step.circuit_flags[j] {
                    circuit_flags[j][step_index] = F::one();
                }
            }
        }

        Self {
            chunks_x: chunks_x
                .into_iter()
                .map(|vals| DensePolynomial::new(vals))
                .collect(),
            chunks_y: chunks_y
                .into_iter()
                .map(|vals| DensePolynomial::new(vals))
                .collect(),
            circuit_flags: circuit_flags
                .into_iter()
                .map(|vals| DensePolynomial::new(vals))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            aux: None,
        }
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct R1CSProof<const C: usize, I: ConstraintInput, F: JoltField> {
    pub key: UniformSpartanKey<C, I, F>,
    pub proof: UniformSpartanProof<C, I, F>,
}

impl<const C: usize, I: ConstraintInput, F: JoltField> R1CSProof<C, I, F> {
    #[tracing::instrument(skip_all, name = "R1CSProof::verify")]
    pub fn verify(&self, transcript: &mut ProofTranscript) -> Result<(), SpartanError> {
        self.proof.verify_precommitted(&self.key, transcript)
    }
}

pub trait ConstraintInput:
    Clone + Copy + Debug + PartialEq + Eq + PartialOrd + Ord + Hash + Sync + Send + 'static
{
    // TODO(moodlezoup): Move flattened version to r1cs preprocesing
    fn flatten<const C: usize>() -> Vec<Self>;
    fn num_inputs<const C: usize>() -> usize {
        Self::flatten::<C>().len()
    }
    fn from_index<const C: usize>(index: usize) -> Self {
        Self::flatten::<C>()[index]
    }
    fn to_index<const C: usize>(&self) -> usize {
        match Self::flatten::<C>().iter().position(|x| x == self) {
            Some(index) => index,
            None => panic!("Invalid JoltIn variant {:?}", self),
        }
    }

    fn get_poly_ref<'a, F: JoltField, PCS: CommitmentScheme<Field = F>>(
        &self,
        jolt_polynomials: &'a JoltPolynomials<F, PCS>,
    ) -> &'a DensePolynomial<F>;

    fn get_poly_ref_mut<F: JoltField, PCS: CommitmentScheme<Field = F>>(
        &self,
        jolt_polynomials: &mut JoltPolynomials<F, PCS>,
    ) -> &mut DensePolynomial<F>;
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Hash, Ord, EnumIter)]
pub enum JoltIn {
    Bytecode_A, // Virtual address
    // Bytecode_V
    Bytecode_ELFAddress,
    Bytecode_Bitflags,
    Bytecode_RS1,
    Bytecode_RS2,
    Bytecode_RD,
    Bytecode_Imm,

    RAM_A,
    // Ram_V
    RS1_Read,
    RS2_Read,
    RD_Read,
    RAM_Read(usize),
    RD_Write,
    RAM_Write(usize),

    ChunksQuery(usize),
    LookupOutput,
    ChunksX(usize),
    ChunksY(usize),

    OpFlags(CircuitFlags),
    InstructionFlags(RV32I),
    Aux(AuxVariable),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Hash, Ord, Default, EnumIter)]
pub enum AuxVariable {
    #[default] // Need a default so that we can derive EnumIter on `JoltIn`
    LeftLookupOperand,
    RightLookupOperand,
    ImmSigned,
    Product,
    RelevantYChunk(usize),
    WriteLookupOutputToRD,
    WritePCtoRD,
    NextPCJump,
    ShouldBranch,
    NextPC,
}

impl_r1cs_input_lc_conversions!(JoltIn, 4);
impl ConstraintInput for JoltIn {
    fn flatten<const C: usize>() -> Vec<Self> {
        JoltIn::iter()
            .flat_map(|variant| match variant {
                Self::RAM_Read(_) => (0..RAM_OPS_PER_INSTRUCTION)
                    .into_iter()
                    .map(|i| Self::RAM_Read(i))
                    .collect(),
                Self::RAM_Write(_) => (0..RAM_OPS_PER_INSTRUCTION)
                    .into_iter()
                    .map(|i| Self::RAM_Write(i))
                    .collect(),
                Self::ChunksQuery(_) => (0..C).into_iter().map(|i| Self::ChunksQuery(i)).collect(),
                Self::ChunksX(_) => (0..C).into_iter().map(|i| Self::ChunksX(i)).collect(),
                Self::ChunksY(_) => (0..C).into_iter().map(|i| Self::ChunksY(i)).collect(),
                Self::OpFlags(_) => CircuitFlags::iter()
                    .map(|flag| Self::OpFlags(flag))
                    .collect(),
                Self::InstructionFlags(_) => RV32I::iter()
                    .map(|flag| Self::InstructionFlags(flag))
                    .collect(),
                Self::Aux(_) => AuxVariable::iter()
                    .flat_map(|aux| match aux {
                        AuxVariable::RelevantYChunk(_) => (0..C)
                            .into_iter()
                            .map(|i| Self::Aux(AuxVariable::RelevantYChunk(i)))
                            .collect(),
                        _ => vec![Self::Aux(aux)],
                    })
                    .collect(),
                _ => vec![variant],
            })
            .collect()
    }
    fn get_poly_ref<'a, F: JoltField, PCS: CommitmentScheme<Field = F>>(
        &self,
        jolt_polynomials: &'a JoltPolynomials<F, PCS>,
    ) -> &'a DensePolynomial<F> {
        todo!()
    }

    fn get_poly_ref_mut<F: JoltField, PCS: CommitmentScheme<Field = F>>(
        &self,
        jolt_polynomials: &mut JoltPolynomials<F, PCS>,
    ) -> &mut DensePolynomial<F> {
        todo!();
    }
}
