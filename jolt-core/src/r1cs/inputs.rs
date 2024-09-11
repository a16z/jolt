#![allow(
    clippy::len_without_is_empty,
    clippy::type_complexity,
    clippy::too_many_arguments
)]

use crate::impl_r1cs_input_lc_conversions;
use crate::jolt::instruction::JoltInstructionSet;
use crate::jolt::vm::rv32i_vm::RV32I;
use crate::jolt::vm::{JoltPolynomials, JoltTraceStep};
use crate::lasso::memory_checking::StructuredPolynomialData;
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

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct AuxVariableStuff<T: CanonicalSerialize + CanonicalDeserialize> {
    pub left_lookup_operand: T,
    pub right_lookup_operand: T,
    pub imm_signed: T,
    pub product: T,
    pub relevant_y_chunks: Vec<T>,
    pub write_lookup_output_to_rd: T,
    pub write_pc_to_rd: T,
    pub next_pc_jump: T,
    pub should_branch: T,
    pub next_pc: T,
}

impl<T: CanonicalSerialize + CanonicalDeserialize + Default> Default for AuxVariableStuff<T> {
    fn default() -> Self {
        Self {
            left_lookup_operand: T::default(),
            right_lookup_operand: T::default(),
            imm_signed: T::default(),
            product: T::default(),
            relevant_y_chunks: todo!(),
            write_lookup_output_to_rd: T::default(),
            write_pc_to_rd: T::default(),
            next_pc_jump: T::default(),
            should_branch: T::default(),
            next_pc: T::default(),
        }
    }
}

impl<T: CanonicalSerialize + CanonicalDeserialize> StructuredPolynomialData<T>
    for AuxVariableStuff<T>
{
    fn read_write_values(&self) -> Vec<&T> {
        let mut values = vec![
            &self.left_lookup_operand,
            &self.right_lookup_operand,
            &self.imm_signed,
            &self.product,
        ];
        values.extend(self.relevant_y_chunks.iter());
        values.extend([
            &self.write_lookup_output_to_rd,
            &self.write_pc_to_rd,
            &self.next_pc_jump,
            &self.should_branch,
            &self.next_pc,
        ]);
        values
    }

    fn init_final_values(&self) -> Vec<&T> {
        vec![]
    }

    fn read_write_values_mut(&mut self) -> Vec<&mut T> {
        let mut values = vec![
            &mut self.left_lookup_operand,
            &mut self.right_lookup_operand,
            &mut self.imm_signed,
            &mut self.product,
        ];
        values.extend(self.relevant_y_chunks.iter_mut());
        values.extend([
            &mut self.write_lookup_output_to_rd,
            &mut self.write_pc_to_rd,
            &mut self.next_pc_jump,
            &mut self.should_branch,
            &mut self.next_pc,
        ]);
        values
    }

    fn init_final_values_mut(&mut self) -> Vec<&mut T> {
        vec![]
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct R1CSStuff<T: CanonicalSerialize + CanonicalDeserialize> {
    pub chunks_x: Vec<T>,
    pub chunks_y: Vec<T>,
    pub circuit_flags: [T; NUM_CIRCUIT_FLAGS],
    pub aux: Option<AuxVariableStuff<T>>,
}

impl<T: CanonicalSerialize + CanonicalDeserialize> StructuredPolynomialData<T> for R1CSStuff<T> {
    fn read_write_values(&self) -> Vec<&T> {
        let aux = self.aux.as_ref().unwrap();
        self.chunks_x
            .iter()
            .chain(self.chunks_y.iter())
            .chain(self.circuit_flags.iter())
            .chain(aux.read_write_values())
            .collect()
    }

    fn init_final_values(&self) -> Vec<&T> {
        vec![]
    }

    fn read_write_values_mut(&mut self) -> Vec<&mut T> {
        let aux = self.aux.as_mut().unwrap();
        self.chunks_x
            .iter_mut()
            .chain(self.chunks_y.iter_mut())
            .chain(self.circuit_flags.iter_mut())
            .chain(aux.read_write_values_mut())
            .collect()
    }

    fn init_final_values_mut(&mut self) -> Vec<&mut T> {
        vec![]
    }
}

pub type R1CSPolynomials<F: JoltField> = R1CSStuff<DensePolynomial<F>>;
pub type R1CSOpenings<F: JoltField> = R1CSStuff<F>;
pub type R1CSCommitments<PCS: CommitmentScheme> = R1CSStuff<PCS::Commitment>;

impl<T: CanonicalSerialize + CanonicalDeserialize + Default> Default for R1CSStuff<T> {
    fn default() -> Self {
        Self {
            chunks_x: todo!(),
            chunks_y: todo!(),
            circuit_flags: std::array::from_fn(|_| T::default()),
            aux: Some(AuxVariableStuff::default()),
        }
    }
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

    fn get_poly_ref<'a, F: JoltField>(
        &self,
        jolt_polynomials: &'a JoltPolynomials<F>,
    ) -> &'a DensePolynomial<F>;

    fn get_poly_ref_mut<'a, F: JoltField>(
        &self,
        jolt_polynomials: &'a mut JoltPolynomials<F>,
    ) -> &'a mut DensePolynomial<F>;
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

    fn get_poly_ref<'a, F: JoltField>(
        &self,
        jolt_polynomials: &'a JoltPolynomials<F>,
    ) -> &'a DensePolynomial<F> {
        let aux_polynomials = jolt_polynomials.r1cs.aux.as_ref().unwrap();
        match self {
            JoltIn::Bytecode_A => &jolt_polynomials.bytecode.a_read_write,
            JoltIn::Bytecode_ELFAddress => &jolt_polynomials.bytecode.v_read_write[0],
            JoltIn::Bytecode_Bitflags => &jolt_polynomials.bytecode.v_read_write[1],
            JoltIn::Bytecode_RD => &jolt_polynomials.bytecode.v_read_write[2],
            JoltIn::Bytecode_RS1 => &jolt_polynomials.bytecode.v_read_write[3],
            JoltIn::Bytecode_RS2 => &jolt_polynomials.bytecode.v_read_write[4],
            JoltIn::Bytecode_Imm => &jolt_polynomials.bytecode.v_read_write[5],
            JoltIn::RAM_A => &jolt_polynomials.read_write_memory.a_ram,
            JoltIn::RS1_Read => &jolt_polynomials.read_write_memory.v_read[0],
            JoltIn::RS2_Read => &jolt_polynomials.read_write_memory.v_read[1],
            JoltIn::RD_Read => &jolt_polynomials.read_write_memory.v_read[2],
            JoltIn::RAM_Read(i) => &jolt_polynomials.read_write_memory.v_read[2 + i],
            JoltIn::RD_Write => &jolt_polynomials.read_write_memory.v_write_rd,
            JoltIn::RAM_Write(i) => &jolt_polynomials.read_write_memory.v_write_ram[*i],
            JoltIn::ChunksQuery(i) => &jolt_polynomials.instruction_lookups.dim[*i],
            JoltIn::LookupOutput => &jolt_polynomials.instruction_lookups.lookup_outputs,
            JoltIn::ChunksX(i) => &jolt_polynomials.r1cs.chunks_x[*i],
            JoltIn::ChunksY(i) => &jolt_polynomials.r1cs.chunks_y[*i],
            JoltIn::OpFlags(i) => &jolt_polynomials.r1cs.circuit_flags[*i as usize],
            JoltIn::InstructionFlags(i) => {
                &jolt_polynomials.instruction_lookups.instruction_flags[RV32I::enum_index(i)]
            }
            Self::Aux(aux) => match aux {
                AuxVariable::LeftLookupOperand => &aux_polynomials.left_lookup_operand,
                AuxVariable::RightLookupOperand => &aux_polynomials.right_lookup_operand,
                AuxVariable::ImmSigned => &aux_polynomials.imm_signed,
                AuxVariable::Product => &aux_polynomials.product,
                AuxVariable::RelevantYChunk(i) => &aux_polynomials.relevant_y_chunks[*i],
                AuxVariable::WriteLookupOutputToRD => &aux_polynomials.write_lookup_output_to_rd,
                AuxVariable::WritePCtoRD => &aux_polynomials.write_pc_to_rd,
                AuxVariable::NextPCJump => &aux_polynomials.next_pc_jump,
                AuxVariable::ShouldBranch => &aux_polynomials.should_branch,
                AuxVariable::NextPC => &aux_polynomials.next_pc,
            },
        }
    }

    fn get_poly_ref_mut<'a, F: JoltField>(
        &self,
        jolt_polynomials: &'a mut JoltPolynomials<F>,
    ) -> &'a mut DensePolynomial<F> {
        let aux_polynomials = jolt_polynomials.r1cs.aux.as_mut().unwrap();
        match self {
            Self::Aux(aux) => match aux {
                AuxVariable::LeftLookupOperand => &mut aux_polynomials.left_lookup_operand,
                AuxVariable::RightLookupOperand => &mut aux_polynomials.right_lookup_operand,
                AuxVariable::ImmSigned => &mut aux_polynomials.imm_signed,
                AuxVariable::Product => &mut aux_polynomials.product,
                AuxVariable::RelevantYChunk(i) => &mut aux_polynomials.relevant_y_chunks[*i],
                AuxVariable::WriteLookupOutputToRD => {
                    &mut aux_polynomials.write_lookup_output_to_rd
                }
                AuxVariable::WritePCtoRD => &mut aux_polynomials.write_pc_to_rd,
                AuxVariable::NextPCJump => &mut aux_polynomials.next_pc_jump,
                AuxVariable::ShouldBranch => &mut aux_polynomials.should_branch,
                AuxVariable::NextPC => &mut aux_polynomials.next_pc,
            },
            _ => panic!("get_poly_ref_mut should only be invoked when computing aux polynomials"),
        }
    }
}
