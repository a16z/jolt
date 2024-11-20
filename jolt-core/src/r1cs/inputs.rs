#![allow(
    clippy::len_without_is_empty,
    clippy::type_complexity,
    clippy::too_many_arguments
)]

use crate::impl_r1cs_input_lc_conversions;
use crate::jolt::instruction::JoltInstructionSet;
use crate::jolt::vm::rv32i_vm::RV32I;
use crate::jolt::vm::{JoltCommitments, JoltStuff, JoltTraceStep};
use crate::lasso::memory_checking::{Initializable, StructuredPolynomialData};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::opening_proof::VerifierOpeningAccumulator;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::utils::transcript::Transcript;

use super::key::UniformSpartanKey;
use super::spartan::{SpartanError, UniformSpartanProof};

use crate::field::JoltField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::log2;
use common::rv_trace::{CircuitFlags, NUM_CIRCUIT_FLAGS};
use std::fmt::Debug;
use std::marker::PhantomData;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

/// Auxiliary variables defined in Jolt's R1CS constraints.
#[derive(Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct AuxVariableStuff<T: CanonicalSerialize + CanonicalDeserialize> {
    pub left_lookup_operand: T,
    pub right_lookup_operand: T,
    pub product: T,
    pub relevant_y_chunks: Vec<T>,
    pub write_lookup_output_to_rd: T,
    pub write_pc_to_rd: T,
    pub next_pc_jump: T,
    pub should_branch: T,
    pub next_pc: T,
}

impl<T: CanonicalSerialize + CanonicalDeserialize + Default> Initializable<T, usize>
    for AuxVariableStuff<T>
{
    #[allow(clippy::field_reassign_with_default)]
    fn initialize(C: &usize) -> Self {
        let mut result = Self::default();
        result.relevant_y_chunks = std::iter::repeat_with(|| T::default()).take(*C).collect();
        result
    }
}

impl<T: CanonicalSerialize + CanonicalDeserialize> StructuredPolynomialData<T>
    for AuxVariableStuff<T>
{
    fn read_write_values(&self) -> Vec<&T> {
        let mut values = vec![
            &self.left_lookup_operand,
            &self.right_lookup_operand,
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

    fn read_write_values_mut(&mut self) -> Vec<&mut T> {
        let mut values = vec![
            &mut self.left_lookup_operand,
            &mut self.right_lookup_operand,
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
}

#[derive(Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct R1CSStuff<T: CanonicalSerialize + CanonicalDeserialize> {
    pub chunks_x: Vec<T>,
    pub chunks_y: Vec<T>,
    pub circuit_flags: [T; NUM_CIRCUIT_FLAGS],
    pub aux: AuxVariableStuff<T>,
}

impl<T: CanonicalSerialize + CanonicalDeserialize + Default> Initializable<T, usize>
    for R1CSStuff<T>
{
    fn initialize(C: &usize) -> Self {
        Self {
            chunks_x: std::iter::repeat_with(|| T::default()).take(*C).collect(),
            chunks_y: std::iter::repeat_with(|| T::default()).take(*C).collect(),
            circuit_flags: std::array::from_fn(|_| T::default()),
            aux: AuxVariableStuff::initialize(C),
        }
    }
}

impl<T: CanonicalSerialize + CanonicalDeserialize> StructuredPolynomialData<T> for R1CSStuff<T> {
    fn read_write_values(&self) -> Vec<&T> {
        self.chunks_x
            .iter()
            .chain(self.chunks_y.iter())
            .chain(self.circuit_flags.iter())
            .chain(self.aux.read_write_values())
            .collect()
    }

    fn read_write_values_mut(&mut self) -> Vec<&mut T> {
        self.chunks_x
            .iter_mut()
            .chain(self.chunks_y.iter_mut())
            .chain(self.circuit_flags.iter_mut())
            .chain(self.aux.read_write_values_mut())
            .collect()
    }
}

/// Witness polynomials specific to Jolt's R1CS constraints (i.e. not used
/// for any offline memory-checking instances).
///
/// Note –– F: JoltField bound is not enforced.
/// See issue #112792 <https://github.com/rust-lang/rust/issues/112792>.
/// Adding #![feature(lazy_type_alias)] to the crate attributes seem to break
/// `alloy_sol_types`.
pub type R1CSPolynomials<F: JoltField> = R1CSStuff<DensePolynomial<F>>;
/// Openings specific to Jolt's R1CS constraints (i.e. not used
/// for any offline memory-checking instances).
///
/// Note –– F: JoltField bound is not enforced.
/// See issue #112792 <https://github.com/rust-lang/rust/issues/112792>.
/// Adding #![feature(lazy_type_alias)] to the crate attributes seem to break
/// `alloy_sol_types`.
pub type R1CSOpenings<F: JoltField> = R1CSStuff<F>;
/// Commitments specific to Jolt's R1CS constraints (i.e. not used
/// for any offline memory-checking instances).
///
/// Note –– PCS: CommitmentScheme bound is not enforced.
/// See issue #112792 <https://github.com/rust-lang/rust/issues/112792>.
/// Adding #![feature(lazy_type_alias)] to the crate attributes seem to break
/// `alloy_sol_types`.
pub type R1CSCommitments<PCS: CommitmentScheme<ProofTranscript>, ProofTranscript: Transcript> =
    R1CSStuff<PCS::Commitment>;

impl<F: JoltField> R1CSPolynomials<F> {
    #[tracing::instrument(skip_all, name = "R1CSPolynomials::new")]
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
            // Actual aux variable polynomials will be computed afterwards
            aux: AuxVariableStuff::initialize(&C),
        }
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct R1CSProof<const C: usize, I: ConstraintInput, F: JoltField, ProofTranscript: Transcript>
{
    pub key: UniformSpartanKey<C, I, F>,
    pub proof: UniformSpartanProof<C, I, F, ProofTranscript>,
    pub _marker: PhantomData<ProofTranscript>,
}

impl<const C: usize, I: ConstraintInput, F: JoltField, ProofTranscript: Transcript>
    R1CSProof<C, I, F, ProofTranscript>
{
    #[tracing::instrument(skip_all, name = "R1CSProof::verify")]
    pub fn verify<PCS>(
        &self,
        commitments: &JoltCommitments<PCS, ProofTranscript>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), SpartanError>
    where
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
    {
        self.proof
            .verify(&self.key, commitments, opening_accumulator, transcript)
    }
}

/// Jolt's R1CS constraint inputs are typically represneted as an enum.
/// This trait serves two main purposes:
/// - Defines a canonical ordering over inputs (and thus indices for each input).
///   This is needed for sumcheck.
/// - Defines a mapping between inputs and Jolt's polynomial/commitment/opening types
///   (i.e. `JoltStuff<T>`).
pub trait ConstraintInput: Clone + Copy + Debug + PartialEq + Sync + Send + 'static {
    /// Returns a flat vector of all unique constraint inputs.
    /// This also serves as a canonical ordering over the inputs.
    fn flatten<const C: usize>() -> Vec<Self>;

    /// The total number of unique constraint inputs
    fn num_inputs<const C: usize>() -> usize {
        Self::flatten::<C>().len()
    }

    /// Converts an index to the corresponding constraint input.
    fn from_index<const C: usize>(index: usize) -> Self {
        Self::flatten::<C>()[index]
    }

    /// Converts a constraint input to its index in the canonical
    /// ordering over inputs given by `ConstraintInput::flatten`.
    fn to_index<const C: usize>(&self) -> usize {
        match Self::flatten::<C>().iter().position(|x| x == self) {
            Some(index) => index,
            None => panic!("Invalid variant {:?}", self),
        }
    }

    /// Gets an immutable reference to a Jolt polynomial/commitment/opening
    /// corresponding to the given constraint input.
    fn get_ref<'a, T: CanonicalSerialize + CanonicalDeserialize + Sync>(
        &self,
        jolt_stuff: &'a JoltStuff<T>,
    ) -> &'a T;

    /// Gets a mutable reference to a Jolt polynomial/commitment/opening
    /// corresponding to the given constraint input.
    fn get_ref_mut<'a, T: CanonicalSerialize + CanonicalDeserialize + Sync>(
        &self,
        jolt_stuff: &'a mut JoltStuff<T>,
    ) -> &'a mut T;
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, EnumIter)]
pub enum JoltR1CSInputs {
    Bytecode_A, // Virtual address
    // Bytecode_V
    Bytecode_ELFAddress,
    Bytecode_Bitflags,
    Bytecode_RS1,
    Bytecode_RS2,
    Bytecode_RD,
    Bytecode_Imm,

    RAM_Address,
    RS1_Read,
    RS2_Read,
    RD_Read,
    RAM_Read,
    RD_Write,
    RAM_Write,

    ChunksQuery(usize),
    LookupOutput,
    ChunksX(usize),
    ChunksY(usize),

    OpFlags(CircuitFlags),
    InstructionFlags(RV32I),
    Aux(AuxVariable),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, EnumIter)]
pub enum AuxVariable {
    #[default] // Need a default so that we can derive EnumIter on `JoltR1CSInputs`
    LeftLookupOperand,
    RightLookupOperand,
    Product,
    RelevantYChunk(usize),
    WriteLookupOutputToRD,
    WritePCtoRD,
    NextPCJump,
    ShouldBranch,
    NextPC,
}

impl_r1cs_input_lc_conversions!(JoltR1CSInputs, 4);
impl ConstraintInput for JoltR1CSInputs {
    fn flatten<const C: usize>() -> Vec<Self> {
        JoltR1CSInputs::iter()
            .flat_map(|variant| match variant {
                Self::ChunksQuery(_) => (0..C).map(Self::ChunksQuery).collect(),
                Self::ChunksX(_) => (0..C).map(Self::ChunksX).collect(),
                Self::ChunksY(_) => (0..C).map(Self::ChunksY).collect(),
                Self::OpFlags(_) => CircuitFlags::iter().map(Self::OpFlags).collect(),
                Self::InstructionFlags(_) => RV32I::iter().map(Self::InstructionFlags).collect(),
                Self::Aux(_) => AuxVariable::iter()
                    .flat_map(|aux| match aux {
                        AuxVariable::RelevantYChunk(_) => (0..C)
                            .map(|i| Self::Aux(AuxVariable::RelevantYChunk(i)))
                            .collect(),
                        _ => vec![Self::Aux(aux)],
                    })
                    .collect(),
                _ => vec![variant],
            })
            .collect()
    }

    fn get_ref<'a, T: CanonicalSerialize + CanonicalDeserialize + Sync>(
        &self,
        jolt: &'a JoltStuff<T>,
    ) -> &'a T {
        let aux_polynomials = &jolt.r1cs.aux;
        match self {
            JoltR1CSInputs::Bytecode_A => &jolt.bytecode.a_read_write,
            JoltR1CSInputs::Bytecode_ELFAddress => &jolt.bytecode.v_read_write[0],
            JoltR1CSInputs::Bytecode_Bitflags => &jolt.bytecode.v_read_write[1],
            JoltR1CSInputs::Bytecode_RD => &jolt.bytecode.v_read_write[2],
            JoltR1CSInputs::Bytecode_RS1 => &jolt.bytecode.v_read_write[3],
            JoltR1CSInputs::Bytecode_RS2 => &jolt.bytecode.v_read_write[4],
            JoltR1CSInputs::Bytecode_Imm => &jolt.bytecode.v_read_write[5],
            JoltR1CSInputs::RAM_Address => &jolt.read_write_memory.a_ram,
            JoltR1CSInputs::RS1_Read => &jolt.read_write_memory.v_read_rs1,
            JoltR1CSInputs::RS2_Read => &jolt.read_write_memory.v_read_rs2,
            JoltR1CSInputs::RD_Read => &jolt.read_write_memory.v_read_rd,
            JoltR1CSInputs::RAM_Read => &jolt.read_write_memory.v_read_ram,
            JoltR1CSInputs::RD_Write => &jolt.read_write_memory.v_write_rd,
            JoltR1CSInputs::RAM_Write => &jolt.read_write_memory.v_write_ram,
            JoltR1CSInputs::ChunksQuery(i) => &jolt.instruction_lookups.dim[*i],
            JoltR1CSInputs::LookupOutput => &jolt.instruction_lookups.lookup_outputs,
            JoltR1CSInputs::ChunksX(i) => &jolt.r1cs.chunks_x[*i],
            JoltR1CSInputs::ChunksY(i) => &jolt.r1cs.chunks_y[*i],
            JoltR1CSInputs::OpFlags(i) => &jolt.r1cs.circuit_flags[*i as usize],
            JoltR1CSInputs::InstructionFlags(i) => {
                &jolt.instruction_lookups.instruction_flags[RV32I::enum_index(i)]
            }
            Self::Aux(aux) => match aux {
                AuxVariable::LeftLookupOperand => &aux_polynomials.left_lookup_operand,
                AuxVariable::RightLookupOperand => &aux_polynomials.right_lookup_operand,
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

    fn get_ref_mut<'a, T: CanonicalSerialize + CanonicalDeserialize + Sync>(
        &self,
        jolt: &'a mut JoltStuff<T>,
    ) -> &'a mut T {
        let aux_polynomials = &mut jolt.r1cs.aux;
        match self {
            Self::Aux(aux) => match aux {
                AuxVariable::LeftLookupOperand => &mut aux_polynomials.left_lookup_operand,
                AuxVariable::RightLookupOperand => &mut aux_polynomials.right_lookup_operand,
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
            _ => panic!("get_ref_mut should only be invoked when computing aux polynomials"),
        }
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;

    use crate::jolt::vm::JoltPolynomials;

    use super::*;

    #[test]
    fn from_index_to_index() {
        const C: usize = 4;
        for i in 0..JoltR1CSInputs::num_inputs::<C>() {
            assert_eq!(i, JoltR1CSInputs::from_index::<C>(i).to_index::<C>());
        }
        for var in JoltR1CSInputs::flatten::<C>() {
            assert_eq!(
                var,
                JoltR1CSInputs::from_index::<C>(JoltR1CSInputs::to_index::<C>(&var))
            );
        }
    }

    #[test]
    fn get_ref() {
        const C: usize = 4;
        let mut jolt_polys: JoltPolynomials<Fr> = JoltPolynomials::default();
        jolt_polys.r1cs = R1CSPolynomials::initialize(&C);

        for aux in AuxVariable::iter().flat_map(|aux| match aux {
            AuxVariable::RelevantYChunk(_) => (0..C)
                .into_iter()
                .map(|i| JoltR1CSInputs::Aux(AuxVariable::RelevantYChunk(i)))
                .collect(),
            _ => vec![JoltR1CSInputs::Aux(aux)],
        }) {
            let ref_ptr = aux.get_ref(&jolt_polys) as *const DensePolynomial<Fr>;
            let ref_mut_ptr = aux.get_ref_mut(&mut jolt_polys) as *const DensePolynomial<Fr>;
            assert_eq!(ref_ptr, ref_mut_ptr, "Pointer mismatch for {:?}", aux);
        }
    }

    #[test]
    fn r1cs_stuff_ordering() {
        const C: usize = 4;
        R1CSOpenings::<Fr>::test_ordering_consistency(&C);
    }
}
