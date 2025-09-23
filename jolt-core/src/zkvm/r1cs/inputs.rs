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
use ark_ff::biginteger::{S128, S64};

use crate::field::JoltField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::XLEN;
use rayon::prelude::*;
use std::fmt::Debug;
use std::marker::PhantomData;
use tracer::instruction::Cycle;

use strum::IntoEnumIterator;

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
    pub trace: &'a [Cycle],
}

impl<'a, F: JoltField, PCS: CommitmentScheme<Field = F>> TraceWitnessAccessor<'a, F, PCS> {
    /// Construct an accessor that borrows `preprocessing` and `trace`.
    pub fn new(preprocessing: &'a JoltProverPreprocessing<F, PCS>, trace: &'a [Cycle]) -> Self {
        Self {
            preprocessing,
            trace,
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

    /// Implementation that returns field elements directly
    #[inline]
    fn value_at_field(&self, input_index: JoltR1CSInputs, t: usize) -> F {
        let len = self.trace.len();
        let get = |idx: usize| -> &Cycle { &self.trace[idx] };
        match input_index {
            JoltR1CSInputs::PC => {
                (self.preprocessing.shared.bytecode.get_pc(get(t)) as u64).to_field()
            }
            JoltR1CSInputs::NextPC => {
                if t + 1 < len {
                    (self.preprocessing.shared.bytecode.get_pc(get(t + 1)) as u64).to_field()
                } else {
                    F::zero()
                }
            }
            JoltR1CSInputs::UnexpandedPC => {
                (get(t).instruction().normalize().address as u64).to_field()
            }
            JoltR1CSInputs::NextUnexpandedPC => {
                if t + 1 < len {
                    (get(t + 1).instruction().normalize().address as u64).to_field()
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
                v.to_field()
            }
            JoltR1CSInputs::RamWriteValue => {
                let v = match get(t).ram_access() {
                    tracer::instruction::RAMAccess::Read(read) => read.value,
                    tracer::instruction::RAMAccess::Write(write) => write.post_value,
                    tracer::instruction::RAMAccess::NoOp => 0,
                };
                v.to_field()
            }
            JoltR1CSInputs::LeftInstructionInput => {
                let (left, _right) = LookupQuery::<XLEN>::to_instruction_inputs(get(t));
                left.to_field()
            }
            JoltR1CSInputs::RightInstructionInput => {
                let (_left, right) = LookupQuery::<XLEN>::to_instruction_inputs(get(t));
                right.to_field()
            }
            JoltR1CSInputs::LeftLookupOperand => {
                let (l, _r) = LookupQuery::<XLEN>::to_lookup_operands(get(t));
                l.to_field()
            }
            JoltR1CSInputs::RightLookupOperand => {
                let (_l, r) = LookupQuery::<XLEN>::to_lookup_operands(get(t));
                r.to_field()
            }
            JoltR1CSInputs::Product => {
                let (left, right) = LookupQuery::<XLEN>::to_instruction_inputs(get(t));
                F::from_i128(right).mul_u64(left)
            }
            JoltR1CSInputs::WriteLookupOutputToRD => {
                let flag = get(t).instruction().circuit_flags()
                    [CircuitFlags::WriteLookupOutputToRD as usize];
                if flag {
                    get(t).rd_write().0.to_field()
                } else {
                    F::zero()
                }
            }
            JoltR1CSInputs::WritePCtoRD => {
                let flag = get(t).instruction().circuit_flags()[CircuitFlags::Jump as usize];
                if flag {
                    get(t).rd_write().0.to_field()
                } else {
                    F::zero()
                }
            }
            JoltR1CSInputs::LookupOutput => {
                F::from_u64(LookupQuery::<XLEN>::to_lookup_output(get(t)))
            }
            JoltR1CSInputs::NextIsNoop => {
                if t + 1 < len {
                    let no = get(t + 1).instruction().circuit_flags()[CircuitFlags::IsNoop];
                    F::from_bool(no)
                } else {
                    F::zero()
                }
            }
            JoltR1CSInputs::ShouldBranch => {
                let is_branch = get(t).instruction().circuit_flags()[CircuitFlags::Branch as usize];
                if is_branch {
                    let out = LookupQuery::<XLEN>::to_lookup_output(get(t));
                    out.to_field()
                } else {
                    F::zero()
                }
            }
            JoltR1CSInputs::ShouldJump => {
                let is_jump = get(t).instruction().circuit_flags()[CircuitFlags::Jump];
                let next_noop = if t + 1 < len {
                    get(t + 1).instruction().circuit_flags()[CircuitFlags::IsNoop]
                } else {
                    true
                };
                F::from_bool(is_jump && !next_noop)
            }
            JoltR1CSInputs::OpFlags(flag) => {
                F::from_bool(get(t).instruction().circuit_flags()[flag as usize])
            }
        }
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
                other,
                input_index.to_index()
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
                other,
                input_index.to_index()
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
                    self.preprocessing
                        .shared
                        .bytecode
                        .get_pc(&self.trace[t + 1]) as u64
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
                let is_branch =
                    self.trace[t].instruction().circuit_flags()[CircuitFlags::Branch as usize];
                if is_branch {
                    LookupQuery::<XLEN>::to_lookup_output(&self.trace[t])
                } else {
                    0
                }
            }
            other => panic!(
                "value_at_u64 called on non-u64 input {:?} (index {})",
                other,
                input_index.to_index()
            ),
        }
    }

    #[inline]
    fn value_at_s64(&self, input_index: JoltR1CSInputs, t: usize) -> S64 {
        match input_index {
            JoltR1CSInputs::Imm => {
                let v = self.trace[t].instruction().normalize().operands.imm;
                let mag = v.unsigned_abs();
                debug_assert!(
                    mag <= u64::MAX as u128,
                    "value_at_s64 overflow for Imm at row {t}: |{v}| > 2^64-1"
                );
                S64::from_u64_with_sign(mag as u64, v >= 0)
            }
            JoltR1CSInputs::RightInstructionInput => {
                let (_l, r) = LookupQuery::<XLEN>::to_instruction_inputs(&self.trace[t]);
                let mag = r.unsigned_abs();
                debug_assert!(
                    mag <= u64::MAX as u128,
                    "value_at_s64 overflow for RightInstructionInput at row {t}: |{r}| > 2^64-1"
                );
                S64::from_u64_with_sign(mag as u64, r >= 0)
            }
            other => panic!(
                "value_at_s64 called on unsupported input {:?} (index {})",
                other,
                input_index.to_index()
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
                other,
                input_index.to_index()
            ),
        }
    }

    #[inline]
    fn value_at_s128(&self, input_index: JoltR1CSInputs, t: usize) -> S128 {
        match input_index {
            JoltR1CSInputs::Product => {
                let (left_u64, right_i128) =
                    LookupQuery::<XLEN>::to_instruction_inputs(&self.trace[t]);
                let left: S64 = S64::from_u64(left_u64);
                let right: S128 = S128::from_i128(right_i128);
                left.mul_trunc::<2, 2>(&right)
            }
            other => panic!(
                "value_at_s128 called on non-signed-128-bit input {:?} (index {})",
                other,
                input_index.to_index()
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
                let next_cycle = if has_next {
                    Some(&accessor.trace[t + 1])
                } else {
                    None
                };
                let next_is_noop = next_cycle
                    .map(|c| c.instruction().circuit_flags()[CircuitFlags::IsNoop])
                    .unwrap_or(false);

                // Instruction inputs and lookup operands
                let (left_u64, right_i128) = LookupQuery::<XLEN>::to_instruction_inputs(cycle);
                // Compute product as S128 (u64 × s64 -> s128) using type-level helpers
                let left_s64: S64 = S64::from_u64(left_u64);
                let right_s128: S128 = S128::from_i128(right_i128);
                let product_s128: S128 = left_s64.mul_trunc::<2, 2>(&right_s128);
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

                // 0: LeftInstructionInput (u64)
                chunk_result[JoltR1CSInputs::LeftInstructionInput.to_index()] +=
                    left_u64.field_mul(*eq_rx_t);
                // 1: RightInstructionInput (i128, really a s64)
                chunk_result[JoltR1CSInputs::RightInstructionInput.to_index()] +=
                    right_i128.field_mul(*eq_rx_t);
                // 2: Product = left_u64 * right_i128
                chunk_result[JoltR1CSInputs::Product.to_index()] +=
                    product_s128.field_mul(*eq_rx_t);
                // 3: WriteLookupOutputToRD = rd if flag else 0 (u8)
                if flags[CircuitFlags::WriteLookupOutputToRD] {
                    chunk_result[JoltR1CSInputs::WriteLookupOutputToRD.to_index()] +=
                        rd.field_mul(*eq_rx_t);
                };
                // 4: WritePCtoRD = rd if Jump else 0 (u8)
                if flags[CircuitFlags::Jump] {
                    chunk_result[JoltR1CSInputs::WritePCtoRD.to_index()] += rd.field_mul(*eq_rx_t);
                };
                // 5: ShouldBranch = LookupOutput if Branch else 0 (u64)
                if flags[CircuitFlags::Branch] {
                    chunk_result[JoltR1CSInputs::ShouldBranch.to_index()] +=
                        lookup_out_u64.field_mul(*eq_rx_t);
                }
                // 6: PC (u64)
                chunk_result[JoltR1CSInputs::PC.to_index()] += pc_u64.field_mul(*eq_rx_t);
                // 7: UnexpandedPC (u64)
                chunk_result[JoltR1CSInputs::UnexpandedPC.to_index()] +=
                    unexp_pc_u64.field_mul(*eq_rx_t);
                // 8: Rd (u8)
                chunk_result[JoltR1CSInputs::Rd.to_index()] += rd.field_mul(*eq_rx_t);
                // 9: Imm (i128)
                chunk_result[JoltR1CSInputs::Imm.to_index()] +=
                    norm.operands.imm.field_mul(*eq_rx_t);
                // 10: RamAddress (u64)
                let ram_addr_u64 = cycle.ram_access().address() as u64;
                chunk_result[JoltR1CSInputs::RamAddress.to_index()] +=
                    ram_addr_u64.field_mul(*eq_rx_t);
                // 11: Rs1Value (u64)
                let rs1_u64 = cycle.rs1_read().1;
                chunk_result[JoltR1CSInputs::Rs1Value.to_index()] += rs1_u64.field_mul(*eq_rx_t);
                // 12: Rs2Value (u64)
                let rs2_u64 = cycle.rs2_read().1;
                chunk_result[JoltR1CSInputs::Rs2Value.to_index()] += rs2_u64.field_mul(*eq_rx_t);
                // 13: RdWriteValue (u64)
                chunk_result[JoltR1CSInputs::RdWriteValue.to_index()] += rdw.field_mul(*eq_rx_t);
                // 14: RamReadValue (u64)
                chunk_result[JoltR1CSInputs::RamReadValue.to_index()] +=
                    ram_rd_u64.field_mul(*eq_rx_t);
                // 15: RamWriteValue (u64)
                chunk_result[JoltR1CSInputs::RamWriteValue.to_index()] +=
                    ram_wr_u64.field_mul(*eq_rx_t);
                // 16: LeftLookupOperand (u64)
                chunk_result[JoltR1CSInputs::LeftLookupOperand.to_index()] +=
                    ll_u64.field_mul(*eq_rx_t);
                // 17: RightLookupOperand (u128)
                chunk_result[JoltR1CSInputs::RightLookupOperand.to_index()] +=
                    rl_u128.field_mul(*eq_rx_t);
                // 18: NextUnexpandedPC (u64)
                chunk_result[JoltR1CSInputs::NextUnexpandedPC.to_index()] +=
                    next_unexp_pc_u64.field_mul(*eq_rx_t);
                // 19: NextPC (u64)
                chunk_result[JoltR1CSInputs::NextPC.to_index()] += next_pc_u64.field_mul(*eq_rx_t);
                // 20: LookupOutput (u64)
                chunk_result[JoltR1CSInputs::LookupOutput.to_index()] +=
                    lookup_out_u64.field_mul(*eq_rx_t);
                // 21: NextIsNoop (bool)
                if next_is_noop {
                    chunk_result[JoltR1CSInputs::NextIsNoop.to_index()] += *eq_rx_t;
                }
                // 22: ShouldJump = Jump && !NextIsNoop (bool)
                if flags[CircuitFlags::Jump] && !next_is_noop {
                    chunk_result[JoltR1CSInputs::ShouldJump.to_index()] += *eq_rx_t;
                }

                // 23..40: OpFlags (bool)
                for flag in CircuitFlags::iter() {
                    if flags[flag] {
                        chunk_result[JoltR1CSInputs::OpFlags(flag).to_index()] += *eq_rx_t;
                    }
                }

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

/// Straightforward reference implementation using a generic row accessor with field-based access.
/// Loops over all `JoltR1CSInputs` and accumulates Σ_t eq(r_cycle, t) * P_i(t).
#[cfg(test)]
pub fn compute_claimed_witness_evals_generic<
    F: JoltField,
    A: WitnessRowAccessor<F, JoltR1CSInputs>,
>(
    r_cycle: &[F],
    accessor: &A,
) -> Vec<F> {
    let eq_rx = EqPolynomial::evals(r_cycle);
    debug_assert_eq!(eq_rx.len(), accessor.num_steps());

    let num_chunks = rayon::current_num_threads().next_power_of_two();
    let chunk_size = (eq_rx.len() / num_chunks).max(1);

    eq_rx
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, eq_chunk)| {
            let mut chunk_result = [F::zero(); NUM_R1CS_INPUTS];
            let mut t = chunk_index * chunk_size;
            for w in eq_chunk {
                for input in ALL_R1CS_INPUTS {
                    let idx = input.to_index();
                    let v = accessor.value_at_field(input, t);
                    chunk_result[idx] += v * *w;
                }
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
    trace: &[Cycle],
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::tracked_ark::TrackedFr;
    use crate::poly::commitment::mock::MockCommitScheme;
    use crate::utils::math::Math;
    use crate::zkvm::{Jolt, JoltRV64IMAC};
    use common::jolt_device::{MemoryConfig, MemoryLayout};
    use tracer::instruction::{Cycle, Instruction};

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

    #[test]
    fn claimed_witness_evals_generic_matches_optimized() {
        // Minimal bytecode; keep empty so test-only PC mapper path is used
        let bytecode: Vec<Instruction> = vec![];

        // Build a trivial memory layout (program_size must be set)
        let mem_layout = MemoryLayout::new(&MemoryConfig {
            program_size: Some(64),
            ..Default::default()
        });

        // Use RV64IMAC implementation to construct shared preprocessing
        let shared = JoltRV64IMAC::shared_preprocess(
            bytecode.clone(),
            mem_layout.clone(),
            vec![],
        );
        let preprocessing = crate::zkvm::JoltProverPreprocessing::<TrackedFr, MockCommitScheme<TrackedFr>> {
            generators: <MockCommitScheme<TrackedFr> as crate::poly::commitment::commitment_scheme::CommitmentScheme>::setup_prover(8),
            shared,
        };

        // Create a tiny trace of pure no-ops and pad to power of two
        let mut trace: Vec<Cycle> = vec![Cycle::NoOp; 3];
        let padded_len = trace.len().next_power_of_two();
        trace.resize(padded_len, Cycle::NoOp);

        // Accessor over this trace
        let accessor = TraceWitnessAccessor::new(&preprocessing, &trace);

        // Choose a random r_cycle of correct dimension
        let r_cycle: Vec<_> = (0..padded_len.log_2())
            .map(|i| TrackedFr::from_u64((i as u64) + 3))
            .collect();

        // Compute both versions
        let fast = compute_claimed_witness_evals(&r_cycle, &accessor);
        let slow = compute_claimed_witness_evals_generic(&r_cycle, &accessor);

        assert_eq!(fast.len(), slow.len());
        for (i, (a, b)) in fast.iter().zip(slow.iter()).enumerate() {
            assert_eq!(*a, *b, "Mismatch at input index {}", i);
        }
    }
}
