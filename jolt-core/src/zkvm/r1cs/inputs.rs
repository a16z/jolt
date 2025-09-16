#![allow(
    clippy::len_without_is_empty,
    clippy::type_complexity,
    clippy::too_many_arguments
)]

use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::opening_proof::{OpeningId, SumcheckId};
use crate::transcripts::Transcript;
use crate::zkvm::instruction::{CircuitFlags, InstructionFlags, LookupQuery};
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use crate::zkvm::JoltProverPreprocessing;

use super::key::UniformSpartanKey;
use super::spartan::UniformSpartanProof;
use crate::utils::small_scalar::SmallScalar;
use ark_ff::biginteger::{S64, S128};
use ark_ff::SignedBigInt;

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
    OpFlags(CircuitFlags),
}

const NUM_R1CS_INPUTS: usize = ALL_R1CS_INPUTS.len();
/// This const serves to define a canonical ordering over inputs (and thus indices
/// for each input). This is needed for sumcheck.
pub const ALL_R1CS_INPUTS: [JoltR1CSInputs; 41] = [
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
pub const COMMITTED_R1CS_INPUTS: [JoltR1CSInputs; 7] = [
    JoltR1CSInputs::LeftInstructionInput,
    JoltR1CSInputs::RightInstructionInput,
    JoltR1CSInputs::Product,
    JoltR1CSInputs::WriteLookupOutputToRD,
    JoltR1CSInputs::WritePCtoRD,
    JoltR1CSInputs::ShouldBranch,
    JoltR1CSInputs::ShouldJump,
];

impl JoltR1CSInputs {
    /// The total number of unique constraint inputs
    pub const fn num_inputs() -> usize {
        NUM_R1CS_INPUTS
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
            JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsRs1Value) => 23,
            JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsRs2Value) => 24,
            JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsPC) => 25,
            JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsImm) => 26,
            JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands) => 27,
            JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands) => 28,
            JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands) => 29,
            JoltR1CSInputs::OpFlags(CircuitFlags::Load) => 30,
            JoltR1CSInputs::OpFlags(CircuitFlags::Store) => 31,
            JoltR1CSInputs::OpFlags(CircuitFlags::Jump) => 32,
            JoltR1CSInputs::OpFlags(CircuitFlags::Branch) => 33,
            JoltR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToRD) => 34,
            JoltR1CSInputs::OpFlags(CircuitFlags::InlineSequenceInstruction) => 35,
            JoltR1CSInputs::OpFlags(CircuitFlags::Assert) => 36,
            JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC) => 37,
            JoltR1CSInputs::OpFlags(CircuitFlags::Advice) => 38,
            JoltR1CSInputs::OpFlags(CircuitFlags::IsNoop) => 39,
            JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed) => 40,
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

/// Read-only, thread-safe accessor for witness values at a given step without
/// materializing full `MultilinearPolynomial`s. Implementations should be
/// zero-copy and cheap per call.
pub trait WitnessRowAccessor<F: JoltField, Index: Copy + Debug>: Send + Sync {
    /// Return the number of steps in the trace
    fn num_steps(&self) -> usize;

    /// Legacy field accessor: default to new semantics; overridden by implementations
    /// that provide an old-field view (e.g., TraceWitnessAccessor::value_at_old).
    fn value_at_field(&self, input_index: Index, t: usize) -> F;

    /// Returns a boolean (must only be called on boolean-valued inputs).
    fn value_at_bool(&self, input_index: Index, t: usize) -> bool;

    /// Returns a u8 (must only be called on u8-valued inputs).
    fn value_at_u8(&self, input_index: Index, t: usize) -> u8;

    /// Returns a u64 (must only be called on u64-valued inputs).
    fn value_at_u64(&self, input_index: Index, t: usize) -> u64;

    /// Returns a S64 (64-bit signed magnitude). Panics if value does not fit in 64 bits.
    fn value_at_s64(&self, input_index: Index, t: usize) -> S64;

    /// Returns a u128 (must only be called on u128-valued inputs).
    fn value_at_u128(&self, input_index: Index, t: usize) -> u128;

    /// Returns a S128 (128-bit signed magnitude). Panics on non-s128 inputs.
    fn value_at_s128(&self, input_index: Index, t: usize) -> S128;
}

/// Lightweight, zero-copy witness accessor backed by `preprocessing` and `trace`.
/// Lifetime `'a` ties this accessor to the borrowed memory.
pub struct TraceWitnessAccessor<'a, F: JoltField, PCS: CommitmentScheme<Field = F>> {
    pub preprocessing: &'a JoltProverPreprocessing<F, PCS>,
    pub trace: &'a [RV32IMCycle],
}

impl<'a, F: JoltField, PCS: CommitmentScheme<Field = F>> TraceWitnessAccessor<'a, F, PCS> {
    /// Construct an accessor that borrows `preprocessing` and `trace`.
    pub fn new(
        preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        trace: &'a [RV32IMCycle],
    ) -> Self {
        Self {
            preprocessing,
            trace,
        }
    }

    /// Old implementation from rv64 branch that returns field elements directly
    #[inline]
    pub fn value_at_old(&self, input_index: JoltR1CSInputs, t: usize) -> F {
        let len = self.trace.len();
        let get = |idx: usize| -> &RV32IMCycle { &self.trace[idx] };
        match input_index {
            JoltR1CSInputs::PC => {
                F::from_u64(self.preprocessing.shared.bytecode.get_pc(get(t)) as u64)
            }
            JoltR1CSInputs::NextPC => {
                if t + 1 < len {
                    F::from_u64(self.preprocessing.shared.bytecode.get_pc(get(t + 1)) as u64)
                } else {
                    F::zero()
                }
            }
            JoltR1CSInputs::UnexpandedPC => {
                F::from_u64(get(t).instruction().normalize().address as u64)
            }
            JoltR1CSInputs::NextUnexpandedPC => {
                if t + 1 < len {
                    F::from_u64(get(t + 1).instruction().normalize().address as u64)
                } else {
                    F::zero()
                }
            }
            JoltR1CSInputs::Rd => F::from_u8(get(t).rd_write().0),
            JoltR1CSInputs::Imm => F::from_i128(get(t).instruction().normalize().operands.imm),
            JoltR1CSInputs::RamAddress => F::from_u64(get(t).ram_access().address() as u64),
            JoltR1CSInputs::Rs1Value => F::from_u64(get(t).rs1_read().1),
            JoltR1CSInputs::Rs2Value => F::from_u64(get(t).rs2_read().1),
            JoltR1CSInputs::RdWriteValue => F::from_u64(get(t).rd_write().2),
            JoltR1CSInputs::RamReadValue => {
                let v = match get(t).ram_access() {
                    tracer::instruction::RAMAccess::Read(read) => read.value,
                    tracer::instruction::RAMAccess::Write(write) => write.pre_value,
                    tracer::instruction::RAMAccess::NoOp => 0,
                };
                F::from_u64(v)
            }
            JoltR1CSInputs::RamWriteValue => {
                let v = match get(t).ram_access() {
                    tracer::instruction::RAMAccess::Read(read) => read.value,
                    tracer::instruction::RAMAccess::Write(write) => write.post_value,
                    tracer::instruction::RAMAccess::NoOp => 0,
                };
                F::from_u64(v)
            }
            JoltR1CSInputs::LeftInstructionInput => {
                let (left, _right) = LookupQuery::<XLEN>::to_instruction_inputs(get(t));
                F::from_u64(left)
            }
            JoltR1CSInputs::RightInstructionInput => {
                let (_left, right) = LookupQuery::<XLEN>::to_instruction_inputs(get(t));
                F::from_i128(right)
            }
            JoltR1CSInputs::LeftLookupOperand => {
                let (l, _r) = LookupQuery::<XLEN>::to_lookup_operands(get(t));
                F::from_u64(l)
            }
            JoltR1CSInputs::RightLookupOperand => {
                let (_l, r) = LookupQuery::<XLEN>::to_lookup_operands(get(t));
                F::from_u128(r)
            }
            JoltR1CSInputs::Product => {
                let (left, right) = LookupQuery::<XLEN>::to_instruction_inputs(get(t));
                F::from_u64(left) * F::from_i128(right)
            }
            JoltR1CSInputs::WriteLookupOutputToRD => {
                let flag = get(t).instruction().circuit_flags()
                    [CircuitFlags::WriteLookupOutputToRD as usize];
                F::from_u8(get(t).rd_write().0 * (flag as u8))
            }
            JoltR1CSInputs::WritePCtoRD => {
                let flag = get(t).instruction().circuit_flags()[CircuitFlags::Jump as usize];
                F::from_u8(get(t).rd_write().0 * (flag as u8))
            }
            JoltR1CSInputs::LookupOutput => {
                F::from_u64(LookupQuery::<XLEN>::to_lookup_output(get(t)))
            }
            JoltR1CSInputs::NextIsNoop => {
                if t + 1 < len {
                    let no = get(t + 1).instruction().circuit_flags()[CircuitFlags::IsNoop];
                    F::from_u8(no as u8)
                } else {
                    F::zero()
                }
            }
            JoltR1CSInputs::ShouldBranch => {
                let is_branch = get(t).instruction().circuit_flags()[CircuitFlags::Branch as usize];
                let out = LookupQuery::<XLEN>::to_lookup_output(get(t)) as u8;
                F::from_u8(out * (is_branch as u8))
            }
            JoltR1CSInputs::ShouldJump => {
                let is_jump = get(t).instruction().circuit_flags()[CircuitFlags::Jump];
                let next_noop = if t + 1 < len {
                    get(t + 1).instruction().circuit_flags()[CircuitFlags::IsNoop]
                } else {
                    true
                };
                F::from_u8((is_jump && !next_noop) as u8)
            }
            JoltR1CSInputs::OpFlags(flag) => {
                F::from_u8(get(t).instruction().circuit_flags()[flag as usize] as u8)
            }
        }
    }
}

impl<'a, F: JoltField, PCS: CommitmentScheme<Field = F>> WitnessRowAccessor<F, JoltR1CSInputs>
    for TraceWitnessAccessor<'a, F, PCS>
{
    #[inline]
    fn num_steps(&self) -> usize {
        self.trace.len()
    }

    #[inline]
    fn value_at_field(&self, input_index: JoltR1CSInputs, t: usize) -> F {
        // Delegate to legacy field evaluation for exact old semantics
        self.value_at_old(input_index, t)
    }

    // ============================
    // Typed fast-path overrides
    // ============================

    #[inline]
    fn value_at_bool(&self, input_index: JoltR1CSInputs, t: usize) -> bool {
        let len = self.trace.len();
        match input_index {
            JoltR1CSInputs::NextIsNoop => {
                if t + 1 < len {
                    self.trace[t + 1].instruction().circuit_flags()[CircuitFlags::IsNoop]
                } else {
                    false
                }
            }
            JoltR1CSInputs::ShouldJump => {
                let is_jump = self.trace[t].instruction().circuit_flags()[CircuitFlags::Jump];
                let next_noop = if t + 1 < len {
                    self.trace[t + 1].instruction().circuit_flags()[CircuitFlags::IsNoop]
                } else {
                    true
                };
                is_jump && !next_noop
            }
            JoltR1CSInputs::OpFlags(flag) => {
                self.trace[t].instruction().circuit_flags()[flag as usize]
            }
            other => panic!(
                "value_at_bool called on non-boolean input {:?} (index {})",
                other, input_index.to_index()
            ),
        }
    }

    #[inline]
    fn value_at_u8(&self, input_index: JoltR1CSInputs, t: usize) -> u8 {
        match input_index {
            JoltR1CSInputs::Rd => self.trace[t].rd_write().0,
            JoltR1CSInputs::WriteLookupOutputToRD => {
                let flag = self.trace[t].instruction().circuit_flags()
                    [CircuitFlags::WriteLookupOutputToRD as usize];
                if flag {
                    self.trace[t].rd_write().0
                } else {
                    0
                }
            }
            JoltR1CSInputs::WritePCtoRD => {
                let flag = self.trace[t].instruction().circuit_flags()[CircuitFlags::Jump as usize];
                if flag {
                    self.trace[t].rd_write().0
                } else {
                    0
                }
            }
            other => panic!(
                "value_at_u8 called on non-u8 input {:?} (index {})",
                other, input_index.to_index()
            ),
        }
    }

    #[inline]
    fn value_at_u64(&self, input_index: JoltR1CSInputs, t: usize) -> u64 {
        let len = self.trace.len();
        match input_index {
            JoltR1CSInputs::PC => self.preprocessing.shared.bytecode.get_pc(&self.trace[t]) as u64,
            JoltR1CSInputs::NextPC => {
                if t + 1 < len {
                    self.preprocessing.shared.bytecode.get_pc(&self.trace[t + 1]) as u64
                } else {
                    0
                }
            }
            JoltR1CSInputs::UnexpandedPC => self.trace[t].instruction().normalize().address as u64,
            JoltR1CSInputs::NextUnexpandedPC => {
                if t + 1 < len {
                    self.trace[t + 1].instruction().normalize().address as u64
                } else {
                    0
                }
            }
            JoltR1CSInputs::RamAddress => self.trace[t].ram_access().address() as u64,
            JoltR1CSInputs::Rs1Value => self.trace[t].rs1_read().1,
            JoltR1CSInputs::Rs2Value => self.trace[t].rs2_read().1,
            JoltR1CSInputs::RdWriteValue => self.trace[t].rd_write().2,
            JoltR1CSInputs::RamReadValue => match self.trace[t].ram_access() {
                tracer::instruction::RAMAccess::Read(read) => read.value,
                tracer::instruction::RAMAccess::Write(write) => write.pre_value,
                tracer::instruction::RAMAccess::NoOp => 0,
            },
            JoltR1CSInputs::RamWriteValue => match self.trace[t].ram_access() {
                tracer::instruction::RAMAccess::Read(read) => read.value,
                tracer::instruction::RAMAccess::Write(write) => write.post_value,
                tracer::instruction::RAMAccess::NoOp => 0,
            },
            JoltR1CSInputs::LeftInstructionInput => {
                let (l, _r) = LookupQuery::<XLEN>::to_instruction_inputs(&self.trace[t]);
                l
            }
            JoltR1CSInputs::LeftLookupOperand => {
                let (l, _r) = LookupQuery::<XLEN>::to_lookup_operands(&self.trace[t]);
                l
            }
            JoltR1CSInputs::LookupOutput => LookupQuery::<XLEN>::to_lookup_output(&self.trace[t]),
            JoltR1CSInputs::ShouldBranch => {
                let is_branch = self.trace[t].instruction().circuit_flags()[CircuitFlags::Branch as usize];
                if is_branch {
                    LookupQuery::<XLEN>::to_lookup_output(&self.trace[t])
                } else {
                    0
                }
            }
            other => panic!(
                "value_at_u64 called on non-u64 input {:?} (index {})",
                other, input_index.to_index()
            ),
        }
    }

    #[inline]
    fn value_at_s64(&self, input_index: JoltR1CSInputs, t: usize) -> S64 {
        match input_index {
            JoltR1CSInputs::Imm => {
                let v = self.trace[t].instruction().normalize().operands.imm;
                let mag = v.unsigned_abs();
                debug_assert!(mag <= u64::MAX as u128,
                    "value_at_s64 overflow for Imm at row {}: |{}| > 2^64-1", t, v);
                SignedBigInt::from_u64_with_sign(mag as u64, v >= 0)
            }
            JoltR1CSInputs::RightInstructionInput => {
                let (_l, r) = LookupQuery::<XLEN>::to_instruction_inputs(&self.trace[t]);
                let mag = r.unsigned_abs();
                debug_assert!(mag <= u64::MAX as u128,
                    "value_at_s64 overflow for RightInstructionInput at row {}: |{}| > 2^64-1", t, r);
                SignedBigInt::from_u64_with_sign(mag as u64, r >= 0)
            }
            other => panic!(
                "value_at_s64 called on unsupported input {:?} (index {})",
                other, input_index.to_index()
            ),
        }
    }

    #[inline]
    fn value_at_u128(&self, input_index: JoltR1CSInputs, t: usize) -> u128 {
        match input_index {
            JoltR1CSInputs::RightLookupOperand => {
                let (_l, r) = LookupQuery::<XLEN>::to_lookup_operands(&self.trace[t]);
                r
            }
            other => panic!(
                "value_at_u128 called on non-u128 input {:?} (index {})",
                other, input_index.to_index()
            ),
        }
    }

    #[inline]
    fn value_at_s128(&self, input_index: JoltR1CSInputs, t: usize) -> S128 {
        match input_index {
            JoltR1CSInputs::Product => {
                let (left_u64, right_i128) = LookupQuery::<XLEN>::to_instruction_inputs(&self.trace[t]);
                let left = SignedBigInt::<1>::from_u64(left_u64);
                let right = SignedBigInt::<2>::from_i128(right_i128);
                left.mul_trunc::<2, 2>(&right)
            }
            other => panic!(
                "value_at_s128 called on non-signed-128-bit input {:?} (index {})",
                other, input_index.to_index()
            ),
        }
    }
}

/// Compute `z(r_cycle) = Σ_t eq(r_cycle, t) * P_i(t)` for all inputs i, without
/// materializing P_i. Returns `[P_0(r_cycle), P_1(r_cycle), ...]` in input order.
#[tracing::instrument(skip_all)]
pub fn compute_claimed_witness_evals<F: JoltField, PCS: CommitmentScheme<Field = F>>(
    r_cycle: &[F],
    accessor: &TraceWitnessAccessor<F, PCS>,
) -> Vec<F> {
    let eq_rx = EqPolynomial::evals(r_cycle);
    let len = accessor.num_steps();

    let num_chunks = rayon::current_num_threads().next_power_of_two();
    let chunk_size = (eq_rx.len() / num_chunks).max(1);

    eq_rx
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, eq_chunk)| {
            let mut chunk_result = [F::zero(); NUM_R1CS_INPUTS];
            let mut t = chunk_index * chunk_size;
            for eq_rx_t in eq_chunk {
                // Row-local cache
                let cycle = &accessor.trace[t];
                let instr = cycle.instruction();
                let flags = instr.circuit_flags();
                let norm = instr.normalize();
                let rd = cycle.rd_write().0 as u8;
                let rdw = cycle.rd_write().2 as u64;

                // Next row cached data
                let has_next = (t + 1) < len;
                let next_cycle = if has_next { Some(&accessor.trace[t + 1]) } else { None };
                let next_is_noop = next_cycle
                    .map(|c| c.instruction().circuit_flags()[CircuitFlags::IsNoop])
                    .unwrap_or(false);

                // Instruction inputs and lookup operands
                let (left_u64, right_i128) = LookupQuery::<XLEN>::to_instruction_inputs(cycle);
                // For streaming eval to field, avoid S128 SmallScalar bounds; compute as i128 directly
                let product_i128 = (left_u64 as i128).saturating_mul(right_i128);
                let (ll_u64, rl_u128) = LookupQuery::<XLEN>::to_lookup_operands(cycle);
                let lookup_out_u64 = LookupQuery::<XLEN>::to_lookup_output(cycle);

                // RAM values
                let (ram_rd_u64, ram_wr_u64) = match cycle.ram_access() {
                    tracer::instruction::RAMAccess::Read(r) => (r.value, r.value),
                    tracer::instruction::RAMAccess::Write(w) => (w.pre_value, w.post_value),
                    tracer::instruction::RAMAccess::NoOp => (0u64, 0u64),
                };

                // PCs
                let pc_u64 = accessor.preprocessing.shared.bytecode.get_pc(cycle) as u64;
                let next_pc_u64 = if let Some(nc) = next_cycle {
                    accessor.preprocessing.shared.bytecode.get_pc(nc) as u64
                } else {
                    0u64
                };
                let unexp_pc_u64 = norm.address as u64;
                let next_unexp_pc_u64 = if let Some(nc) = next_cycle {
                    nc.instruction().normalize().address as u64
                } else {
                    0u64
                };

                // Common flags
                let f_wr_lu_rd = flags[CircuitFlags::WriteLookupOutputToRD];
                let f_jump = flags[CircuitFlags::Jump];
                let f_branch = flags[CircuitFlags::Branch];

                // 0: LeftInstructionInput (u64)
                chunk_result[0] = left_u64.field_mul(*eq_rx_t);
                // 1: RightInstructionInput (i128, really a s64)
                chunk_result[1] = right_i128.field_mul(*eq_rx_t);
                // 2: Product = left_u64 * right_i128
                chunk_result[2] = product_i128.field_mul(*eq_rx_t);
                // 3: WriteLookupOutputToRD = rd if flag else 0 (u8)
                if flags[CircuitFlags::WriteLookupOutputToRD] {
                    chunk_result[3] = rd.field_mul(*eq_rx_t);
                };
                // 4: WritePCtoRD = rd if Jump else 0 (u8)
                if flags[CircuitFlags::Jump] {
                    chunk_result[4] = rd.field_mul(*eq_rx_t);
                };
                // 5: ShouldBranch = LookupOutput if Branch else 0 (u64)
                if flags[CircuitFlags::Branch] {
                    chunk_result[5] = lookup_out_u64.field_mul(*eq_rx_t);
                }
                // 6: PC (u64)
                chunk_result[6] = pc_u64.field_mul(*eq_rx_t);
                // 7: UnexpandedPC (u64)
                chunk_result[7] = unexp_pc_u64.field_mul(*eq_rx_t);
                // 8: Rd (u8)
                chunk_result[8] = rd.field_mul(*eq_rx_t);
                // 9: Imm (i128)
                chunk_result[9] = norm.operands.imm.field_mul(*eq_rx_t);
                // 10: RamAddress (u64)
                let ram_addr_u64 = cycle.ram_access().address() as u64;
                chunk_result[10] = ram_addr_u64.field_mul(*eq_rx_t);
                // 11: Rs1Value (u64)
                let rs1_u64 = cycle.rs1_read().1;
                chunk_result[11] = rs1_u64.field_mul(*eq_rx_t);
                // 12: Rs2Value (u64)
                let rs2_u64 = cycle.rs2_read().1;
                chunk_result[12] = rs2_u64.field_mul(*eq_rx_t);
                // 13: RdWriteValue (u64)
                chunk_result[13] = rdw.field_mul(*eq_rx_t);
                // 14: RamReadValue (u64)
                chunk_result[14] = ram_rd_u64.field_mul(*eq_rx_t);
                // 15: RamWriteValue (u64)
                chunk_result[15] = ram_wr_u64.field_mul(*eq_rx_t);
                // 16: LeftLookupOperand (u64)
                chunk_result[16] = ll_u64.field_mul(*eq_rx_t);
                // 17: RightLookupOperand (u128)
                chunk_result[17] = rl_u128.field_mul(*eq_rx_t);
                // 18: NextUnexpandedPC (u64)
                chunk_result[18] = next_unexp_pc_u64.field_mul(*eq_rx_t);
                // 19: NextPC (u64)
                chunk_result[19] = next_pc_u64.field_mul(*eq_rx_t);
                // 20: LookupOutput (u64)
                chunk_result[20] = lookup_out_u64.field_mul(*eq_rx_t);
                // 21: NextIsNoop (bool)
                if next_is_noop {
                    chunk_result[21] = *eq_rx_t;
                }
                // 22: ShouldJump = Jump && !NextIsNoop (bool)
                if flags[CircuitFlags::Jump] && !next_is_noop {
                    chunk_result[22] = *eq_rx_t;
                }
                // 23..40: OpFlags (bool)
                if flags[CircuitFlags::LeftOperandIsRs1Value] { chunk_result[23] = *eq_rx_t; }
                if flags[CircuitFlags::RightOperandIsRs2Value] { chunk_result[24] = *eq_rx_t; }
                if flags[CircuitFlags::LeftOperandIsPC] { chunk_result[25] = *eq_rx_t; }
                if flags[CircuitFlags::RightOperandIsImm] { chunk_result[26] = *eq_rx_t; }
                if flags[CircuitFlags::AddOperands] { chunk_result[27] = *eq_rx_t; }
                if flags[CircuitFlags::SubtractOperands] { chunk_result[28] = *eq_rx_t; }
                if flags[CircuitFlags::MultiplyOperands] { chunk_result[29] = *eq_rx_t; }
                if flags[CircuitFlags::Load] { chunk_result[30] = *eq_rx_t; }
                if flags[CircuitFlags::Store] { chunk_result[31] = *eq_rx_t; }
                if flags[CircuitFlags::Jump] { chunk_result[32] = *eq_rx_t; }
                if flags[CircuitFlags::Branch] { chunk_result[33] = *eq_rx_t; }
                if flags[CircuitFlags::WriteLookupOutputToRD] { chunk_result[34] = *eq_rx_t; }
                if flags[CircuitFlags::InlineSequenceInstruction] { chunk_result[35] = *eq_rx_t; }
                if flags[CircuitFlags::Assert] { chunk_result[36] = *eq_rx_t; }
                if flags[CircuitFlags::DoNotUpdateUnexpandedPC] { chunk_result[37] = *eq_rx_t; }
                if flags[CircuitFlags::Advice] { chunk_result[38] = *eq_rx_t; }
                if flags[CircuitFlags::IsNoop] { chunk_result[39] = *eq_rx_t; }
                if flags[CircuitFlags::IsCompressed] { chunk_result[40] = *eq_rx_t; }

                t += 1;
            }
            chunk_result
        })
        .reduce(
            || [F::zero(); NUM_R1CS_INPUTS],
            |mut acc, evals| {
                for i in 0..NUM_R1CS_INPUTS {
                    acc[i] += evals[i];
                }
                acc
            },
        )
        .to_vec()
}

/// Single-pass generation of UnexpandedPC(t), PC(t), and IsNoop(t) witnesses.
/// Reduces traversals from three to one for stage-3 PC sumcheck inputs.
#[tracing::instrument(skip_all)]
pub fn generate_pc_noop_witnesses<F, PCS>(
    preprocessing: &JoltProverPreprocessing<F, PCS>,
    trace: &[RV32IMCycle],
) -> (
    MultilinearPolynomial<F>, // UnexpandedPC(t)
    MultilinearPolynomial<F>, // PC(t)
    MultilinearPolynomial<F>, // IsNoop(t)
)
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    let len = trace.len();
    let mut unexpanded_pc: Vec<u64> = vec![0; len];
    let mut pc: Vec<u64> = vec![0; len];
    let mut is_noop: Vec<u8> = vec![0; len];

    unexpanded_pc
        .par_iter_mut()
        .zip(pc.par_iter_mut())
        .zip(is_noop.par_iter_mut())
        .zip(trace.par_iter())
        .for_each(|(((u, p), n), cycle)| {
            *u = cycle.instruction().normalize().address as u64;
            *p = preprocessing.shared.bytecode.get_pc(cycle) as u64;
            *n = cycle.instruction().circuit_flags()[CircuitFlags::IsNoop] as u8;
        });

    (unexpanded_pc.into(), pc.into(), is_noop.into())
}

// =====================================================================================
// Streaming typed evaluation helpers (SVO types) co-located with witness accessor
// =====================================================================================

/// Generic evaluator: evaluates an LC over witness row into an I8OrI96.
/// Uses small signed fast-path and falls back to sign/magnitude accumulation.
// #[inline]
// pub fn eval_az_generic<F: JoltField>(
//     a_lc: &LC,
//     accessor: &dyn WitnessRowAccessor<F, JoltR1CSInputs>,
//     row: usize,
// ) -> I8OrI96 {
//     // Pass 1: small i8 fast path (Bool/U8 inputs only, i8 coeffs, no overflow)
//     let mut acc_i8: i8 = 0;
//     let mut small_ok = true;
//     a_lc.for_each_term(|input_index, coeff| {
//         if !small_ok {
//             return;
//         }
//         // Fast-path requires coeff to be I8; otherwise bail out to generic path.
//         let coeff_i8: i8 = match coeff.is_small {
//             true => coeff.small_i8,
//             false => {
//                 small_ok = false;
//                 return;
//             }
//         };
//         // Only Bool/U8 are allowed to keep the i8 path (rv64 A-side small sums)
//         let sc = accessor.value_at(input_index, row);
//         let v_i8 = match sc {
//             SmallScalar::Bool(b) => {
//                 if b {
//                     1
//                 } else {
//                     0
//                 }
//             }
//             SmallScalar::U8(v) => v as i8,
//             _ => {
//                 small_ok = false;
//                 return;
//             }
//         };
//         let (prod, of1) = v_i8.overflowing_mul(coeff_i8);
//         let (sum, of2) = acc_i8.overflowing_add(prod);
//         acc_i8 = sum;
//         if of1 || of2 {
//             small_ok = false;
//         }
//     });
//     if small_ok {
//         if let Some(cst) = a_lc.const_term() {
//             // For Az constraints, constants are expected to be I8; if not, bail to generic.
//             if cst.is_small {
//                 let c8 = cst.small_i8;
//                 let (sum, of) = acc_i8.overflowing_add(c8);
//                 acc_i8 = sum;
//                 if of {
//                     small_ok = false;
//                 }
//             } else {
//                 small_ok = false;
//             }
//         }
//     }
//     if small_ok {
//         return I8OrI96::I8(acc_i8);
//     }

//     // Pass 2: sign-aware u64 magnitude accumulation (rv64-compatible integer LC semantics)
//     let mut mag: u64 = 0;
//     let mut sign: bool = true; // true => positive
//     a_lc.for_each_term(|input_index, coeff| {
//         let sc = accessor.value_at(input_index, row);
//         let v_i128 = sc.as_i128();
//         let v_mag_u128 = v_i128.unsigned_abs();
//         let v_mag_u64 = if v_mag_u128 > u64::MAX as u128 {
//             u64::MAX
//         } else {
//             v_mag_u128 as u64
//         };
//         let c_i128 = coeff.to_i128();
//         let term_mag = v_mag_u64.saturating_mul(c_i128.unsigned_abs() as u64);
//         let val_pos = v_i128 >= 0;
//         let coeff_pos = c_i128 >= 0;
//         let term_pos = val_pos == coeff_pos; // sign(product) positive if signs equal
//         let (new_mag, new_pos) = add_with_sign_u64(mag, sign, term_mag, term_pos);
//         mag = new_mag;
//         sign = new_pos;
//     });
//     if let Some(cst) = a_lc.const_term() {
//         let cst_i128 = cst.to_i128();
//         let c_mag = if cst_i128.unsigned_abs() > u64::MAX as u128 {
//             u64::MAX
//         } else {
//             cst_i128.unsigned_abs() as u64
//         };
//         let c_pos = cst_i128 >= 0;
//         let (new_mag, new_pos) = add_with_sign_u64(mag, sign, c_mag, c_pos);
//         mag = new_mag;
//         sign = new_pos;
//     }
//     I8OrI96::S64(SignedBigInt::from_u64_with_sign(mag, sign))
// }

// /// Generic evaluator: evaluates an LC over witness row into a S160.
// /// Always accumulates into 192-bit signed magnitude (SignedBigInt<3>).
// #[inline]
// pub fn eval_bz_generic<F: JoltField>(
//     b_lc: &LC,
//     accessor: &dyn WitnessRowAccessor<F, JoltR1CSInputs>,
//     row: usize,
// ) -> S160 {
//     // Try S64 fast path: accumulate using u64 magnitude with sign if all term products fit in u64
//     let mut s64_mag: u64 = 0;
//     let mut s64_pos: bool = true;
//     let mut fits_s64 = true;
//     b_lc.for_each_term(|input_index, coeff| {
//         if !fits_s64 {
//             return;
//         }
//         let sc = accessor.value_at(input_index, row);
//         // Extract magnitude and sign of witness with minimal conversions.
//         let (v_mag_u64, v_nonneg) = match sc {
//             SmallScalar::Bool(b) => (if b { 1u64 } else { 0u64 }, true),
//             SmallScalar::U8(v) => (v as u64, true),
//             SmallScalar::U64(v) => (v, true),
//             SmallScalar::I64(v) => (v.unsigned_abs(), v >= 0),
//             SmallScalar::I128(v) => {
//                 // rv64 semantics: B-side integer LC uses the signed value
//                 let mag = v.unsigned_abs();
//                 if mag > u64::MAX as u128 {
//                     fits_s64 = false;
//                     return;
//                 }
//                 (mag as u64, v >= 0)
//             }
//             SmallScalar::U128(v) => {
//                 if v > u64::MAX as u128 {
//                     fits_s64 = false;
//                     return;
//                 }
//                 (v as u64, true)
//             }
//             SmallScalar::S128(v) => (v.magnitude, v.is_positive),
//         };
//         let (c_mag_u64, c_nonneg) = match coeff {
//             _ => {
//                 let cv = coeff.to_i128();
//                 let c_mag = cv.unsigned_abs();
//                 if c_mag > u64::MAX as u128 {
//                     fits_s64 = false;
//                     return;
//                 }
//                 (c_mag as u64, cv >= 0)
//             }
//         };
//         let prod_mag_u128 = (v_mag_u64 as u128) * (c_mag_u64 as u128);
//         if prod_mag_u128 > u64::MAX as u128 {
//             fits_s64 = false;
//             return;
//         }
//         let prod_mag_u64 = prod_mag_u128 as u64;
//         let term_pos = v_nonneg == c_nonneg;
//         let (new_mag, new_pos) = add_with_sign_u64(s64_mag, s64_pos, prod_mag_u64, term_pos);
//         s64_mag = new_mag;
//         s64_pos = new_pos;
//     });
//     if fits_s64 {
//         if let Some(cst) = b_lc.const_term() {
//             let cv = cst.to_i128();
//             let c_mag = cv.unsigned_abs();
//             if c_mag > u64::MAX as u128 {
//                 fits_s64 = false;
//             } else {
//                 let (new_mag, new_pos) = add_with_sign_u64(s64_mag, s64_pos, c_mag as u64, cv >= 0);
//                 s64_mag = new_mag;
//                 s64_pos = new_pos;
//             }
//         }
//     }
//     if fits_s64 {
//         return S160::S64(SignedBigInt::from_u64_with_sign(s64_mag, s64_pos));
//     }

//     // Fallback: S128 accumulation (rv64 semantics for wider products)
//     let mut acc = SignedBigInt::<2>::zero();
//     b_lc.for_each_term(|input_index, coeff| {
//         let sc = accessor.value_at(input_index, row);
//         // rv64: use signed integer value directly
//         let v_i128 = sc.as_i128();
//         let v = SignedBigInt::<2>::from_i128(v_i128);
//         let c = SignedBigInt::<2>::from_i128(coeff.to_i128());
//         let term = v.mul_trunc::<2, 2>(&c);
//         acc = acc.add(term);
//     });
//     if let Some(cst) = b_lc.const_term() {
//         let c = SignedBigInt::<2>::from_i128(cst.to_i128());
//         acc = acc.add(c);
//     }
//     S160::S128(acc)
// }

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
