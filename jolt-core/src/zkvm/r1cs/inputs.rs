#![allow(
    clippy::len_without_is_empty,
    clippy::type_complexity,
    clippy::too_many_arguments
)]

#[cfg(test)]
use crate::impl_r1cs_input_lc_conversions;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::opening_proof::{OpeningId, SumcheckId};
use crate::transcripts::Transcript;
use crate::zkvm::instruction::{CircuitFlags, InstructionFlags, LookupQuery};
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use crate::zkvm::JoltProverPreprocessing;

use super::key::UniformSpartanKey;
use super::spartan::UniformSpartanProof;

use crate::field::JoltField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::XLEN;
use rayon::prelude::*;
use std::fmt::Debug;
use std::marker::PhantomData;
use tracer::instruction::RV32IMCycle;

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct R1CSProof<F: JoltField, ProofTranscript: Transcript> {
    pub key: UniformSpartanKey<F>,
    pub proof: UniformSpartanProof<F, ProofTranscript>,
    pub _marker: PhantomData<ProofTranscript>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JoltR1CSInputs {
    PC,                    // Virtual (bytecode raf)
    UnexpandedPC,          // Virtual (bytecode rv)
    Rd,                    // Virtual (bytecode rv)
    Imm,                   // Virtual (bytecode rv)
    RamAddress,            // Virtual (RAM raf)
    Rs1Value,              // Virtual (registers rv)
    Rs2Value,              // Virtual (registers rv)
    RdWriteValue,          // Virtual (registers wv)
    RamReadValue,          // Virtual (RAM rv)
    RamWriteValue,         // Virtual (RAM wv)
    LeftInstructionInput,  // to_lookup_query -> to_instruction_operands
    RightInstructionInput, // to_lookup_query -> to_instruction_operands
    LeftLookupOperand,     // Virtual (instruction raf)
    RightLookupOperand,    // Virtual (instruction raf)
    Product,               // LeftInstructionOperand * RightInstructionOperand
    WriteLookupOutputToRD,
    WritePCtoRD,
    ShouldBranch,
    NextUnexpandedPC, // Virtual (spartan shift sumcheck)
    NextPC,           // Virtual (spartan shift sumcheck)
    LookupOutput,     // Virtual (instruction rv)
    NextIsNoop,       // Virtual (spartan shift sumcheck)
    ShouldJump,
    CompressedDoNotUpdateUnexpPC,
    OpFlags(CircuitFlags),
}

/// This const serves to define a canonical ordering over inputs (and thus indices
/// for each input). This is needed for sumcheck.
pub const ALL_R1CS_INPUTS: [JoltR1CSInputs; 42] = [
    JoltR1CSInputs::LeftInstructionInput,
    JoltR1CSInputs::RightInstructionInput,
    JoltR1CSInputs::Product,
    JoltR1CSInputs::WriteLookupOutputToRD,
    JoltR1CSInputs::WritePCtoRD,
    JoltR1CSInputs::ShouldBranch,
    JoltR1CSInputs::PC,
    JoltR1CSInputs::UnexpandedPC,
    JoltR1CSInputs::Rd,
    JoltR1CSInputs::Imm,
    JoltR1CSInputs::RamAddress,
    JoltR1CSInputs::Rs1Value,
    JoltR1CSInputs::Rs2Value,
    JoltR1CSInputs::RdWriteValue,
    JoltR1CSInputs::RamReadValue,
    JoltR1CSInputs::RamWriteValue,
    JoltR1CSInputs::LeftLookupOperand,
    JoltR1CSInputs::RightLookupOperand,
    JoltR1CSInputs::NextUnexpandedPC,
    JoltR1CSInputs::NextPC,
    JoltR1CSInputs::LookupOutput,
    JoltR1CSInputs::NextIsNoop,
    JoltR1CSInputs::ShouldJump,
    JoltR1CSInputs::CompressedDoNotUpdateUnexpPC,
    JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsRs1Value),
    JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsRs2Value),
    JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsPC),
    JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsImm),
    JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands),
    JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands),
    JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands),
    JoltR1CSInputs::OpFlags(CircuitFlags::Load),
    JoltR1CSInputs::OpFlags(CircuitFlags::Store),
    JoltR1CSInputs::OpFlags(CircuitFlags::Jump),
    JoltR1CSInputs::OpFlags(CircuitFlags::Branch),
    JoltR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToRD),
    JoltR1CSInputs::OpFlags(CircuitFlags::InlineSequenceInstruction),
    JoltR1CSInputs::OpFlags(CircuitFlags::Assert),
    JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC),
    JoltR1CSInputs::OpFlags(CircuitFlags::Advice),
    JoltR1CSInputs::OpFlags(CircuitFlags::IsNoop),
    JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed),
];

/// The subset of `ALL_R1CS_INPUTS` that are committed. The rest of
/// the inputs are virtual polynomials.
pub const COMMITTED_R1CS_INPUTS: [JoltR1CSInputs; 8] = [
    JoltR1CSInputs::LeftInstructionInput,
    JoltR1CSInputs::RightInstructionInput,
    JoltR1CSInputs::Product,
    JoltR1CSInputs::WriteLookupOutputToRD,
    JoltR1CSInputs::WritePCtoRD,
    JoltR1CSInputs::ShouldBranch,
    JoltR1CSInputs::ShouldJump,
    JoltR1CSInputs::CompressedDoNotUpdateUnexpPC,
];

impl JoltR1CSInputs {
    /// The total number of unique constraint inputs
    pub fn num_inputs() -> usize {
        ALL_R1CS_INPUTS.len()
    }

    /// Converts an index to the corresponding constraint input.
    pub fn from_index(index: usize) -> Self {
        ALL_R1CS_INPUTS[index]
    }

    /// Converts a constraint input to its index in the canonical
    /// ordering over inputs given by `ALL_R1CS_INPUTS`.
    ///
    /// This is tested to align with ALL_R1CS_INPUTS, and this is the default version
    /// since it is simple pattern matching and not iteration over all r1cs inputs.
    pub const fn to_index(&self) -> usize {
        match self {
            JoltR1CSInputs::LeftInstructionInput => 0,
            JoltR1CSInputs::RightInstructionInput => 1,
            JoltR1CSInputs::Product => 2,
            JoltR1CSInputs::WriteLookupOutputToRD => 3,
            JoltR1CSInputs::WritePCtoRD => 4,
            JoltR1CSInputs::ShouldBranch => 5,
            JoltR1CSInputs::PC => 6,
            JoltR1CSInputs::UnexpandedPC => 7,
            JoltR1CSInputs::Rd => 8,
            JoltR1CSInputs::Imm => 9,
            JoltR1CSInputs::RamAddress => 10,
            JoltR1CSInputs::Rs1Value => 11,
            JoltR1CSInputs::Rs2Value => 12,
            JoltR1CSInputs::RdWriteValue => 13,
            JoltR1CSInputs::RamReadValue => 14,
            JoltR1CSInputs::RamWriteValue => 15,
            JoltR1CSInputs::LeftLookupOperand => 16,
            JoltR1CSInputs::RightLookupOperand => 17,
            JoltR1CSInputs::NextUnexpandedPC => 18,
            JoltR1CSInputs::NextPC => 19,
            JoltR1CSInputs::LookupOutput => 20,
            JoltR1CSInputs::NextIsNoop => 21,
            JoltR1CSInputs::ShouldJump => 22,
            JoltR1CSInputs::CompressedDoNotUpdateUnexpPC => 23,
            JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsRs1Value) => 24,
            JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsRs2Value) => 25,
            JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsPC) => 26,
            JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsImm) => 27,
            JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands) => 28,
            JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands) => 29,
            JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands) => 30,
            JoltR1CSInputs::OpFlags(CircuitFlags::Load) => 31,
            JoltR1CSInputs::OpFlags(CircuitFlags::Store) => 32,
            JoltR1CSInputs::OpFlags(CircuitFlags::Jump) => 33,
            JoltR1CSInputs::OpFlags(CircuitFlags::Branch) => 34,
            JoltR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToRD) => 35,
            JoltR1CSInputs::OpFlags(CircuitFlags::InlineSequenceInstruction) => 36,
            JoltR1CSInputs::OpFlags(CircuitFlags::Assert) => 37,
            JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC) => 38,
            JoltR1CSInputs::OpFlags(CircuitFlags::Advice) => 39,
            JoltR1CSInputs::OpFlags(CircuitFlags::IsNoop) => 40,
            JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed) => 41,
        }
    }

    pub fn generate_witness<F, PCS>(
        &self,
        trace: &[RV32IMCycle],
        preprocessing: &JoltProverPreprocessing<F, PCS>,
    ) -> MultilinearPolynomial<F>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
    {
        match self {
            JoltR1CSInputs::PC => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| preprocessing.shared.bytecode.get_pc(cycle) as u64)
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::NextPC => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .skip(1)
                    .map(|cycle| preprocessing.shared.bytecode.get_pc(cycle) as u64)
                    .chain(rayon::iter::once(0))
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::UnexpandedPC => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| cycle.instruction().normalize().address as u64)
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::Rd => {
                let coeffs: Vec<u8> = trace.par_iter().map(|cycle| cycle.rd_write().0).collect();
                coeffs.into()
            }
            JoltR1CSInputs::Imm => {
                let coeffs: Vec<i128> = trace
                    .par_iter()
                    .map(|cycle| cycle.instruction().normalize().operands.imm)
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::RamAddress => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| cycle.ram_access().address() as u64)
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::Rs1Value => {
                let coeffs: Vec<u64> = trace.par_iter().map(|cycle| cycle.rs1_read().1).collect();
                coeffs.into()
            }
            JoltR1CSInputs::Rs2Value => {
                let coeffs: Vec<u64> = trace.par_iter().map(|cycle| cycle.rs2_read().1).collect();
                coeffs.into()
            }
            JoltR1CSInputs::RdWriteValue => {
                let coeffs: Vec<u64> = trace.par_iter().map(|cycle| cycle.rd_write().2).collect();
                coeffs.into()
            }
            JoltR1CSInputs::RamReadValue => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| match cycle.ram_access() {
                        tracer::instruction::RAMAccess::Read(read) => read.value,
                        tracer::instruction::RAMAccess::Write(write) => write.pre_value,
                        tracer::instruction::RAMAccess::NoOp => 0,
                    })
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::RamWriteValue => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| match cycle.ram_access() {
                        tracer::instruction::RAMAccess::Read(read) => read.value,
                        tracer::instruction::RAMAccess::Write(write) => write.post_value,
                        tracer::instruction::RAMAccess::NoOp => 0,
                    })
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::LeftInstructionInput => {
                CommittedPolynomial::LeftInstructionInput.generate_witness(preprocessing, trace)
            }
            JoltR1CSInputs::RightInstructionInput => {
                CommittedPolynomial::RightInstructionInput.generate_witness(preprocessing, trace)
            }
            JoltR1CSInputs::LeftLookupOperand => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| LookupQuery::<XLEN>::to_lookup_operands(cycle).0)
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::RightLookupOperand => {
                let coeffs: Vec<u128> = trace
                    .par_iter()
                    .map(|cycle| LookupQuery::<XLEN>::to_lookup_operands(cycle).1)
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::Product => {
                CommittedPolynomial::Product.generate_witness(preprocessing, trace)
            }
            JoltR1CSInputs::WriteLookupOutputToRD => {
                CommittedPolynomial::WriteLookupOutputToRD.generate_witness(preprocessing, trace)
            }
            JoltR1CSInputs::WritePCtoRD => {
                CommittedPolynomial::WritePCtoRD.generate_witness(preprocessing, trace)
            }
            JoltR1CSInputs::LookupOutput => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(LookupQuery::<XLEN>::to_lookup_output)
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::NextUnexpandedPC => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .skip(1)
                    .chain(rayon::iter::once(&RV32IMCycle::NoOp))
                    .map(|cycle| cycle.instruction().normalize().address as u64)
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::ShouldBranch => {
                CommittedPolynomial::ShouldBranch.generate_witness(preprocessing, trace)
            }
            JoltR1CSInputs::ShouldJump => {
                CommittedPolynomial::ShouldJump.generate_witness(preprocessing, trace)
            }
            JoltR1CSInputs::NextIsNoop => {
                // TODO(moodlezoup): Boolean polynomial
                let coeffs: Vec<u8> = trace
                    .par_iter()
                    .skip(1)
                    .map(|cycle| cycle.instruction().circuit_flags()[CircuitFlags::IsNoop] as u8)
                    .chain(rayon::iter::once(0))
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::CompressedDoNotUpdateUnexpPC => {
                CommittedPolynomial::CompressedDoNotUpdateUnexpPC
                    .generate_witness(preprocessing, trace)
            }
            JoltR1CSInputs::OpFlags(flag) => {
                // TODO(moodlezoup): Boolean polynomial
                let coeffs: Vec<u8> = trace
                    .par_iter()
                    .map(|cycle| cycle.instruction().circuit_flags()[*flag as usize] as u8)
                    .collect();
                coeffs.into()
            }
        }
    }
}

/// Converts a JoltR1CSInputs to a CommittedPolynomial if the input represents a committed
/// polynomial, and returns an error otherwise.
impl TryFrom<&JoltR1CSInputs> for CommittedPolynomial {
    type Error = &'static str;

    fn try_from(input: &JoltR1CSInputs) -> Result<Self, Self::Error> {
        match input {
            JoltR1CSInputs::LeftInstructionInput => Ok(CommittedPolynomial::LeftInstructionInput),
            JoltR1CSInputs::RightInstructionInput => Ok(CommittedPolynomial::RightInstructionInput),
            JoltR1CSInputs::Product => Ok(CommittedPolynomial::Product),
            JoltR1CSInputs::WriteLookupOutputToRD => Ok(CommittedPolynomial::WriteLookupOutputToRD),
            JoltR1CSInputs::WritePCtoRD => Ok(CommittedPolynomial::WritePCtoRD),
            JoltR1CSInputs::ShouldBranch => Ok(CommittedPolynomial::ShouldBranch),
            JoltR1CSInputs::ShouldJump => Ok(CommittedPolynomial::ShouldJump),
            JoltR1CSInputs::CompressedDoNotUpdateUnexpPC => {
                Ok(CommittedPolynomial::CompressedDoNotUpdateUnexpPC)
            }
            _ => Err("{value} is not a committed polynomial"),
        }
    }
}

/// Converts a JoltR1CSInputs to a VirtualPolynomial if the input represents a virtual polynomial,
/// and returns an error otherwise.
impl TryFrom<&JoltR1CSInputs> for VirtualPolynomial {
    type Error = &'static str;

    fn try_from(input: &JoltR1CSInputs) -> Result<Self, Self::Error> {
        match input {
            JoltR1CSInputs::PC => Ok(VirtualPolynomial::PC),
            JoltR1CSInputs::UnexpandedPC => Ok(VirtualPolynomial::UnexpandedPC),
            JoltR1CSInputs::Rd => Ok(VirtualPolynomial::Rd),
            JoltR1CSInputs::Imm => Ok(VirtualPolynomial::Imm),
            JoltR1CSInputs::RamAddress => Ok(VirtualPolynomial::RamAddress),
            JoltR1CSInputs::Rs1Value => Ok(VirtualPolynomial::Rs1Value),
            JoltR1CSInputs::Rs2Value => Ok(VirtualPolynomial::Rs2Value),
            JoltR1CSInputs::RdWriteValue => Ok(VirtualPolynomial::RdWriteValue),
            JoltR1CSInputs::RamReadValue => Ok(VirtualPolynomial::RamReadValue),
            JoltR1CSInputs::RamWriteValue => Ok(VirtualPolynomial::RamWriteValue),
            JoltR1CSInputs::LeftLookupOperand => Ok(VirtualPolynomial::LeftLookupOperand),
            JoltR1CSInputs::RightLookupOperand => Ok(VirtualPolynomial::RightLookupOperand),
            JoltR1CSInputs::NextUnexpandedPC => Ok(VirtualPolynomial::NextUnexpandedPC),
            JoltR1CSInputs::NextPC => Ok(VirtualPolynomial::NextPC),
            JoltR1CSInputs::NextIsNoop => Ok(VirtualPolynomial::NextIsNoop),
            JoltR1CSInputs::LookupOutput => Ok(VirtualPolynomial::LookupOutput),
            JoltR1CSInputs::OpFlags(flag) => Ok(VirtualPolynomial::OpFlags(*flag)),
            _ => Err("{value} is not a virtual polynomial"),
        }
    }
}

/// Converts a JoltR1CSInputs to an OpeningId by determining if it is a virtual or committed
/// polynomial, returning an error otherwise.
impl TryFrom<&JoltR1CSInputs> for OpeningId {
    type Error = &'static str;

    fn try_from(input: &JoltR1CSInputs) -> Result<Self, Self::Error> {
        if let Ok(poly) = VirtualPolynomial::try_from(input) {
            Ok(OpeningId::Virtual(poly, SumcheckId::SpartanOuter))
        } else if let Ok(poly) = CommittedPolynomial::try_from(input) {
            Ok(OpeningId::Committed(poly, SumcheckId::SpartanOuter))
        } else {
            Err("Could not map {value} to an OpeningId")
        }
    }
}

// Legacy conversions for old_ops, only used in test code
#[cfg(test)]
impl_r1cs_input_lc_conversions!(JoltR1CSInputs);

#[cfg(test)]
mod tests {
    use super::*;

    impl JoltR1CSInputs {
        /// Alternative const implementation that searches through ALL_R1CS_INPUTS array.
        /// This is used for testing to ensure the simple pattern matching to_index()
        /// returns the same results as searching through the array.
        const fn find_index_via_array_search(&self) -> usize {
            let mut i = 0;
            while i < ALL_R1CS_INPUTS.len() {
                if self.const_eq(&ALL_R1CS_INPUTS[i]) {
                    return i;
                }
                i += 1;
            }
            panic!("Invalid variant")
        }

        /// Const-compatible equality check for JoltR1CSInputs
        const fn const_eq(&self, other: &JoltR1CSInputs) -> bool {
            match (self, other) {
                (JoltR1CSInputs::PC, JoltR1CSInputs::PC) => true,
                (JoltR1CSInputs::UnexpandedPC, JoltR1CSInputs::UnexpandedPC) => true,
                (JoltR1CSInputs::Rd, JoltR1CSInputs::Rd) => true,
                (JoltR1CSInputs::Imm, JoltR1CSInputs::Imm) => true,
                (JoltR1CSInputs::RamAddress, JoltR1CSInputs::RamAddress) => true,
                (JoltR1CSInputs::Rs1Value, JoltR1CSInputs::Rs1Value) => true,
                (JoltR1CSInputs::Rs2Value, JoltR1CSInputs::Rs2Value) => true,
                (JoltR1CSInputs::RdWriteValue, JoltR1CSInputs::RdWriteValue) => true,
                (JoltR1CSInputs::RamReadValue, JoltR1CSInputs::RamReadValue) => true,
                (JoltR1CSInputs::RamWriteValue, JoltR1CSInputs::RamWriteValue) => true,
                (JoltR1CSInputs::LeftInstructionInput, JoltR1CSInputs::LeftInstructionInput) => {
                    true
                }
                (JoltR1CSInputs::RightInstructionInput, JoltR1CSInputs::RightInstructionInput) => {
                    true
                }
                (JoltR1CSInputs::LeftLookupOperand, JoltR1CSInputs::LeftLookupOperand) => true,
                (JoltR1CSInputs::RightLookupOperand, JoltR1CSInputs::RightLookupOperand) => true,
                (JoltR1CSInputs::Product, JoltR1CSInputs::Product) => true,
                (JoltR1CSInputs::WriteLookupOutputToRD, JoltR1CSInputs::WriteLookupOutputToRD) => {
                    true
                }
                (JoltR1CSInputs::WritePCtoRD, JoltR1CSInputs::WritePCtoRD) => true,
                (JoltR1CSInputs::ShouldBranch, JoltR1CSInputs::ShouldBranch) => true,
                (JoltR1CSInputs::NextUnexpandedPC, JoltR1CSInputs::NextUnexpandedPC) => true,
                (JoltR1CSInputs::NextPC, JoltR1CSInputs::NextPC) => true,
                (JoltR1CSInputs::LookupOutput, JoltR1CSInputs::LookupOutput) => true,
                (JoltR1CSInputs::NextIsNoop, JoltR1CSInputs::NextIsNoop) => true,
                (JoltR1CSInputs::ShouldJump, JoltR1CSInputs::ShouldJump) => true,
                (
                    JoltR1CSInputs::CompressedDoNotUpdateUnexpPC,
                    JoltR1CSInputs::CompressedDoNotUpdateUnexpPC,
                ) => true,
                (JoltR1CSInputs::OpFlags(flag1), JoltR1CSInputs::OpFlags(flag2)) => {
                    self.const_eq_circuit_flags(*flag1, *flag2)
                }
                _ => false,
            }
        }

        /// Const-compatible equality check for CircuitFlags
        const fn const_eq_circuit_flags(&self, flag1: CircuitFlags, flag2: CircuitFlags) -> bool {
            matches!(
                (flag1, flag2),
                (
                    CircuitFlags::LeftOperandIsRs1Value,
                    CircuitFlags::LeftOperandIsRs1Value
                ) | (
                    CircuitFlags::RightOperandIsRs2Value,
                    CircuitFlags::RightOperandIsRs2Value
                ) | (CircuitFlags::LeftOperandIsPC, CircuitFlags::LeftOperandIsPC)
                    | (
                        CircuitFlags::RightOperandIsImm,
                        CircuitFlags::RightOperandIsImm
                    )
                    | (CircuitFlags::AddOperands, CircuitFlags::AddOperands)
                    | (
                        CircuitFlags::SubtractOperands,
                        CircuitFlags::SubtractOperands
                    )
                    | (
                        CircuitFlags::MultiplyOperands,
                        CircuitFlags::MultiplyOperands
                    )
                    | (CircuitFlags::Load, CircuitFlags::Load)
                    | (CircuitFlags::Store, CircuitFlags::Store)
                    | (CircuitFlags::Jump, CircuitFlags::Jump)
                    | (CircuitFlags::Branch, CircuitFlags::Branch)
                    | (
                        CircuitFlags::WriteLookupOutputToRD,
                        CircuitFlags::WriteLookupOutputToRD
                    )
                    | (
                        CircuitFlags::InlineSequenceInstruction,
                        CircuitFlags::InlineSequenceInstruction
                    )
                    | (CircuitFlags::Assert, CircuitFlags::Assert)
                    | (
                        CircuitFlags::DoNotUpdateUnexpandedPC,
                        CircuitFlags::DoNotUpdateUnexpandedPC
                    )
                    | (CircuitFlags::Advice, CircuitFlags::Advice)
                    | (CircuitFlags::IsNoop, CircuitFlags::IsNoop)
                    | (CircuitFlags::IsCompressed, CircuitFlags::IsCompressed)
            )
        }
    }

    #[test]
    fn to_index_consistency() {
        // Ensure to_index() and find_index_via_array_search() return the same values.
        // This validates that the simple pattern matching in to_index() correctly
        // aligns with the ordering in ALL_R1CS_INPUTS.
        for var in ALL_R1CS_INPUTS {
            assert_eq!(
                var.to_index(),
                var.find_index_via_array_search(),
                "Index mismatch for variant {:?}: pattern_match={}, array_search={}",
                var,
                var.to_index(),
                var.find_index_via_array_search()
            );
        }
    }
}
