#![allow(
    clippy::len_without_is_empty,
    clippy::type_complexity,
    clippy::too_many_arguments
)]

use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::opening_proof::{OpeningId, SumcheckId};
use crate::utils::small_scalar::SmallScalar;
use crate::zkvm::instruction::{CircuitFlags, InstructionFlags, LookupQuery, NUM_CIRCUIT_FLAGS};
use crate::zkvm::spartan::product::VirtualProductType;
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use crate::zkvm::JoltSharedPreprocessing;

use crate::field::JoltField;
use ark_ff::biginteger::{S128, S64};
use common::constants::XLEN;
use rayon::prelude::*;
use std::fmt::Debug;
use tracer::instruction::Cycle;

use strum::IntoEnumIterator;

/// Fully materialized, typed view of all R1CS inputs for a single row (cycle).
/// Filled once and reused to evaluate all constraints without re-reading the trace.
/// Total size: 208 bytes, alignment: 16 bytes
#[derive(Clone, Debug)]
pub struct R1CSCycleInputs {
    /// Left instruction input as a u64 bit-pattern.
    /// Typically `Rs1Value` or the current `UnexpandedPC`, depending on `CircuitFlags`.
    pub left_input: u64,
    /// Right instruction input as signed-magnitude `S64`.
    /// Typically `Imm` or `Rs2Value` with exact integer semantics.
    pub right_input: S64,
    /// Signed-magnitude `S128` product consistent with the `Product` witness.
    /// Computed from `left_input` × `right_input` using the same truncation semantics as the witness.
    pub product: S128,

    /// Left lookup operand (u64) for the instruction lookup query.
    /// Matches `LeftLookupOperand` virtual polynomial semantics.
    pub left_lookup: u64,
    /// Right lookup operand (u128) for the instruction lookup query.
    /// Full-width integer encoding used by add/sub/mul/advice cases.
    pub right_lookup: u128,
    /// Instruction lookup output (u64) for this cycle.
    pub lookup_output: u64,

    /// Destination register index (Rd).
    pub rd_addr: u8,
    /// Value read from Rs1 in this cycle.
    pub rs1_read_value: u64,
    /// Value read from Rs2 in this cycle.
    pub rs2_read_value: u64,
    /// Value written to Rd in this cycle.
    pub rd_write_value: u64,

    /// RAM address accessed this cycle.
    pub ram_addr: u64,
    /// RAM read value for `Read`, pre-write value for `Write`, or 0 for `NoOp`.
    pub ram_read_value: u64,
    /// RAM write value: equals read value for `Read`, post-write value for `Write`, or 0 for `NoOp`.
    pub ram_write_value: u64,

    /// Expanded PC used by bytecode instance.
    pub pc: u64,
    /// Expanded PC for next cycle, or 0 if this is the last cycle in the domain.
    pub next_pc: u64,
    /// Unexpanded PC (normalized instruction address) for this cycle.
    pub unexpanded_pc: u64,
    /// Unexpanded PC for next cycle, or 0 if this is the last cycle in the domain.
    pub next_unexpanded_pc: u64,

    /// Immediate operand as signed-magnitude `S64`.
    pub imm: S64,

    /// Per-instruction circuit flags indexed by `CircuitFlags`.
    pub flags: [bool; NUM_CIRCUIT_FLAGS],
    /// `IsNoop` flag for the next cycle (false for last cycle).
    pub next_is_noop: bool,

    /// Derived: `Jump && !NextIsNoop`.
    pub should_jump: bool,
    /// Derived: `LookupOutput` if `Branch`, else 0.
    pub should_branch: u64,

    /// Rd index if `WriteLookupOutputToRD`, else 0 (u8 domain used as selector).
    pub write_lookup_output_to_rd_addr: u8,
    /// Rd index if `Jump`, else 0 (u8 domain used as selector).
    pub write_pc_to_rd_addr: u8,
}

impl R1CSCycleInputs {
    /// Build directly from the execution trace and preprocessing,
    /// mirroring the optimized semantics used in `compute_claimed_witness_evals`.
    pub fn from_trace<F>(preprocessing: &JoltSharedPreprocessing, trace: &[Cycle], t: usize) -> Self
    where
        F: JoltField,
    {
        let len = trace.len();
        let cycle = &trace[t];
        let instr = cycle.instruction();
        let flags_view = instr.circuit_flags();
        let norm = instr.normalize();

        // Next-cycle context
        let next_cycle = if t + 1 < len {
            Some(&trace[t + 1])
        } else {
            None
        };

        // Instruction inputs and product
        let (left_input, right_i128) = LookupQuery::<XLEN>::to_instruction_inputs(cycle);
        let left_s64: S64 = S64::from_u64(left_input);
        let right_mag = right_i128.unsigned_abs();
        debug_assert!(
            right_mag <= u64::MAX as u128,
            "RightInstructionInput overflow at row {t}: |{right_i128}| > 2^64-1"
        );
        let right_input = S64::from_u64_with_sign(right_mag as u64, right_i128 >= 0);
        let right_s128: S128 = S128::from_i128(right_i128);
        let product: S128 = left_s64.mul_trunc::<2, 2>(&right_s128);

        // Lookup operands and output
        let (left_lookup, right_lookup) = LookupQuery::<XLEN>::to_lookup_operands(cycle);
        let lookup_output = LookupQuery::<XLEN>::to_lookup_output(cycle);

        // Registers
        let rd_addr = cycle.rd_write().0;
        let rs1_read_value = cycle.rs1_read().1;
        let rs2_read_value = cycle.rs2_read().1;
        let rd_write_value = cycle.rd_write().2;

        // RAM
        let ram_addr = cycle.ram_access().address() as u64;
        let (ram_read_value, ram_write_value) = match cycle.ram_access() {
            tracer::instruction::RAMAccess::Read(r) => (r.value, r.value),
            tracer::instruction::RAMAccess::Write(w) => (w.pre_value, w.post_value),
            tracer::instruction::RAMAccess::NoOp => (0u64, 0u64),
        };

        // PCs
        let pc = preprocessing.bytecode.get_pc(cycle) as u64;
        let next_pc = if let Some(nc) = next_cycle {
            preprocessing.bytecode.get_pc(nc) as u64
        } else {
            0u64
        };
        let unexpanded_pc = norm.address as u64;
        let next_unexpanded_pc = if let Some(nc) = next_cycle {
            nc.instruction().normalize().address as u64
        } else {
            0u64
        };

        // Immediate
        let imm_i128 = norm.operands.imm;
        let imm_mag = imm_i128.unsigned_abs();
        debug_assert!(
            imm_mag <= u64::MAX as u128,
            "Imm overflow at row {t}: |{imm_i128}| > 2^64-1"
        );
        let imm = S64::from_u64_with_sign(imm_mag as u64, imm_i128 >= 0);

        // Flags and derived booleans
        let mut flags = [false; NUM_CIRCUIT_FLAGS];
        for flag in CircuitFlags::iter() {
            flags[flag] = flags_view[flag];
        }
        let next_is_noop = if let Some(nc) = next_cycle {
            nc.instruction().circuit_flags()[CircuitFlags::IsNoop]
        } else {
            false
        };
        let should_jump = flags_view[CircuitFlags::Jump] && !next_is_noop;
        let should_branch = if flags_view[CircuitFlags::Branch] {
            lookup_output
        } else {
            0u64
        };

        // Write-to-Rd selectors (masked by flags)
        let write_lookup_output_to_rd_addr = if flags_view[CircuitFlags::WriteLookupOutputToRD] {
            rd_addr
        } else {
            0
        };
        let write_pc_to_rd_addr = if flags_view[CircuitFlags::Jump] {
            rd_addr
        } else {
            0
        };

        Self {
            left_input,
            right_input,
            product,
            left_lookup,
            right_lookup,
            lookup_output,
            rd_addr,
            rs1_read_value,
            rs2_read_value,
            rd_write_value,
            ram_addr,
            ram_read_value,
            ram_write_value,
            pc,
            next_pc,
            unexpanded_pc,
            next_unexpanded_pc,
            imm,
            flags,
            next_is_noop,
            should_jump,
            should_branch,
            write_lookup_output_to_rd_addr,
            write_pc_to_rd_addr,
        }
    }

    /// Get field value for a specific input index (only for testing)
    #[cfg(test)]
    pub fn to_field<F: JoltField>(&self, input_index: JoltR1CSInputs) -> F {
        match input_index {
            JoltR1CSInputs::LeftInstructionInput => self.left_input.to_field(),
            JoltR1CSInputs::RightInstructionInput => F::from_i128(self.right_input.to_i128()),
            JoltR1CSInputs::Product => {
                F::from_i128(self.right_input.to_i128()).mul_u64(self.left_input)
            }
            JoltR1CSInputs::WriteLookupOutputToRD => {
                (self.write_lookup_output_to_rd_addr as u64).to_field()
            }
            JoltR1CSInputs::WritePCtoRD => (self.write_pc_to_rd_addr as u64).to_field(),
            JoltR1CSInputs::ShouldBranch => self.should_branch.to_field(),
            JoltR1CSInputs::PC => self.pc.to_field(),
            JoltR1CSInputs::UnexpandedPC => self.unexpanded_pc.to_field(),
            JoltR1CSInputs::Rd => (self.rd_addr as u64).to_field(),
            JoltR1CSInputs::Imm => F::from_i128(self.imm.to_i128()),
            JoltR1CSInputs::RamAddress => self.ram_addr.to_field(),
            JoltR1CSInputs::Rs1Value => self.rs1_read_value.to_field(),
            JoltR1CSInputs::Rs2Value => self.rs2_read_value.to_field(),
            JoltR1CSInputs::RdWriteValue => self.rd_write_value.to_field(),
            JoltR1CSInputs::RamReadValue => self.ram_read_value.to_field(),
            JoltR1CSInputs::RamWriteValue => self.ram_write_value.to_field(),
            JoltR1CSInputs::LeftLookupOperand => self.left_lookup.to_field(),
            JoltR1CSInputs::RightLookupOperand => self.right_lookup.to_field(),
            JoltR1CSInputs::NextUnexpandedPC => self.next_unexpanded_pc.to_field(),
            JoltR1CSInputs::NextPC => self.next_pc.to_field(),
            JoltR1CSInputs::LookupOutput => self.lookup_output.to_field(),
            JoltR1CSInputs::NextIsNoop => F::from_bool(self.next_is_noop),
            JoltR1CSInputs::ShouldJump => F::from_bool(self.should_jump),
            JoltR1CSInputs::OpFlags(flag) => F::from_bool(self.flags[flag]),
        }
    }
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
pub const COMMITTED_R1CS_INPUTS: [JoltR1CSInputs; 6] = [
    JoltR1CSInputs::LeftInstructionInput,
    JoltR1CSInputs::RightInstructionInput,
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
            JoltR1CSInputs::Product => Ok(VirtualPolynomial::Product),
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

/// Compute `z(r_cycle) = Σ_t eq(r_cycle, t) * P_i(t)` for all inputs i, without
/// materializing P_i. Returns `[P_0(r_cycle), P_1(r_cycle), ...]` in input order.
/// TODO: use delayed reduction while computing the sum
#[tracing::instrument(skip_all)]
pub fn compute_claimed_witness_evals<F: JoltField>(
    preprocessing: &JoltSharedPreprocessing,
    trace: &[Cycle],
    r_cycle: &[F::Challenge],
) -> Vec<F> {
    // Implement double-sum semantics: sum_{x1} eq1[x1] * (sum_{x2} eq2[x2] * term(x1||x2))
    let m = r_cycle.len() / 2;
    let (r2, r1) = r_cycle.split_at(m);
    let (eq_one, eq_two) = rayon::join(|| EqPolynomial::evals(r2), || EqPolynomial::evals(r1));

    (0..eq_one.len())
        .into_par_iter()
        .map(|x1| {
            let eq1_val = eq_one[x1];

            // Inner serial accumulation over x2: accumulate eq2[x2] * P_i(row)
            let mut inner = [F::zero(); NUM_R1CS_INPUTS];
            for x2 in 0..eq_two.len() {
                let eq2_val = eq_two[x2];
                let idx = x1 * eq_two.len() + x2;

                // Materialize row directly from trace and preprocessing
                let row = R1CSCycleInputs::from_trace::<F>(preprocessing, trace, idx);

                // Accumulate directly from materialized row using field_mul on raw values
                inner[JoltR1CSInputs::LeftInstructionInput.to_index()] +=
                    row.left_input.field_mul(eq2_val);
                inner[JoltR1CSInputs::RightInstructionInput.to_index()] +=
                    row.right_input.field_mul(eq2_val);
                inner[JoltR1CSInputs::Product.to_index()] += row.product.field_mul(eq2_val);
                inner[JoltR1CSInputs::WriteLookupOutputToRD.to_index()] +=
                    row.write_lookup_output_to_rd_addr.field_mul(eq2_val);
                inner[JoltR1CSInputs::WritePCtoRD.to_index()] +=
                    row.write_pc_to_rd_addr.field_mul(eq2_val);
                inner[JoltR1CSInputs::ShouldBranch.to_index()] +=
                    row.should_branch.field_mul(eq2_val);
                inner[JoltR1CSInputs::PC.to_index()] += row.pc.field_mul(eq2_val);
                inner[JoltR1CSInputs::UnexpandedPC.to_index()] +=
                    row.unexpanded_pc.field_mul(eq2_val);
                inner[JoltR1CSInputs::Rd.to_index()] += row.rd_addr.field_mul(eq2_val);
                inner[JoltR1CSInputs::Imm.to_index()] += row.imm.to_i128().field_mul(eq2_val);
                inner[JoltR1CSInputs::RamAddress.to_index()] += row.ram_addr.field_mul(eq2_val);
                inner[JoltR1CSInputs::Rs1Value.to_index()] += row.rs1_read_value.field_mul(eq2_val);
                inner[JoltR1CSInputs::Rs2Value.to_index()] += row.rs2_read_value.field_mul(eq2_val);
                inner[JoltR1CSInputs::RdWriteValue.to_index()] +=
                    row.rd_write_value.field_mul(eq2_val);
                inner[JoltR1CSInputs::RamReadValue.to_index()] +=
                    row.ram_read_value.field_mul(eq2_val);
                inner[JoltR1CSInputs::RamWriteValue.to_index()] +=
                    row.ram_write_value.field_mul(eq2_val);
                inner[JoltR1CSInputs::LeftLookupOperand.to_index()] +=
                    row.left_lookup.field_mul(eq2_val);
                inner[JoltR1CSInputs::RightLookupOperand.to_index()] +=
                    row.right_lookup.field_mul(eq2_val);
                inner[JoltR1CSInputs::NextUnexpandedPC.to_index()] +=
                    row.next_unexpanded_pc.field_mul(eq2_val);
                inner[JoltR1CSInputs::NextPC.to_index()] += row.next_pc.field_mul(eq2_val);
                inner[JoltR1CSInputs::LookupOutput.to_index()] +=
                    row.lookup_output.field_mul(eq2_val);
                if row.next_is_noop {
                    inner[JoltR1CSInputs::NextIsNoop.to_index()] += eq2_val;
                }
                if row.should_jump {
                    inner[JoltR1CSInputs::ShouldJump.to_index()] += eq2_val;
                }
                for flag in CircuitFlags::iter() {
                    if row.flags[flag] {
                        inner[JoltR1CSInputs::OpFlags(flag).to_index()] += eq2_val;
                    }
                }
            }

            // Now multiply accumulated inner sums by eq1[x1]
            for i in 0..NUM_R1CS_INPUTS {
                inner[i] *= eq1_val;
            }
            inner
        })
        .reduce(
            || [F::zero(); NUM_R1CS_INPUTS],
            |mut acc, item| {
                for i in 0..NUM_R1CS_INPUTS {
                    acc[i] += item[i];
                }
                acc
            },
        )
        .to_vec()
}

/// Single-pass generation of UnexpandedPC(t), PC(t), and IsNoop(t) witnesses.
/// Reduces traversals from three to one for stage-3 PC sumcheck inputs.
#[tracing::instrument(skip_all)]
pub fn generate_pc_noop_witnesses<F>(
    preprocessing: &JoltSharedPreprocessing,
    trace: &[Cycle],
) -> (
    MultilinearPolynomial<F>, // UnexpandedPC(t)
    MultilinearPolynomial<F>, // PC(t)
    MultilinearPolynomial<F>, // IsNoop(t)
)
where
    F: JoltField,
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
            *p = preprocessing.bytecode.get_pc(cycle) as u64;
            *n = cycle.instruction().circuit_flags()[CircuitFlags::IsNoop] as u8;
        });

    (unexpanded_pc.into(), pc.into(), is_noop.into())
}

#[tracing::instrument(skip_all)]
pub fn generate_product_virtualization_witnesses<F>(
    trace: &[Cycle],
) -> (
    MultilinearPolynomial<F>, // LeftInstructionInput(t)
    MultilinearPolynomial<F>, // RightInstructionInput(t)
)
where
    F: JoltField,
{
    let len = trace.len();
    let mut left_input: Vec<u64> = vec![0; len];
    let mut right_input: Vec<i128> = vec![0; len];

    left_input
        .par_iter_mut()
        .zip(right_input.par_iter_mut())
        .zip(trace.par_iter())
        .for_each(|((left, right), cycle)| {
            (*left, *right) = LookupQuery::<XLEN>::to_instruction_inputs(cycle);
        });

    (left_input.into(), right_input.into())
}

pub fn generate_virtual_product_witnesses<F>(
    product_type: VirtualProductType,
    trace: &[Cycle],
) -> (
    MultilinearPolynomial<F>, // Left polynomial
    MultilinearPolynomial<F>, // Right polynomial
)
where
    F: JoltField,
{
    let len = trace.len();

    match product_type {
        VirtualProductType::Instruction => generate_product_virtualization_witnesses(trace),
        VirtualProductType::WriteLookupOutputToRD => {
            let mut rd_addrs: Vec<u8> = vec![0; len];
            let mut flags: Vec<u8> = vec![0; len];

            rd_addrs
                .par_iter_mut()
                .zip(flags.par_iter_mut())
                .zip(trace.par_iter())
                .for_each(|((rd, flag), cycle)| {
                    *rd = cycle.rd_write().0;
                    *flag = cycle.instruction().circuit_flags()[CircuitFlags::WriteLookupOutputToRD]
                        as u8;
                });

            (rd_addrs.into(), flags.into())
        }
        VirtualProductType::WritePCtoRD => {
            let mut rd_addrs: Vec<u8> = vec![0; len];
            let mut flags: Vec<u8> = vec![0; len];

            rd_addrs
                .par_iter_mut()
                .zip(flags.par_iter_mut())
                .zip(trace.par_iter())
                .for_each(|((rd, flag), cycle)| {
                    *rd = cycle.rd_write().0;
                    *flag = cycle.instruction().circuit_flags()[CircuitFlags::Jump] as u8;
                });

            (rd_addrs.into(), flags.into())
        }
        VirtualProductType::ShouldBranch => {
            let mut lookup_outputs: Vec<u64> = vec![0; len];
            let mut flags: Vec<u8> = vec![0; len];

            lookup_outputs
                .par_iter_mut()
                .zip(flags.par_iter_mut())
                .zip(trace.par_iter())
                .for_each(|((output, flag), cycle)| {
                    *output = LookupQuery::<XLEN>::to_lookup_output(cycle);
                    *flag = cycle.instruction().circuit_flags()[CircuitFlags::Branch] as u8;
                });

            (lookup_outputs.into(), flags.into())
        }
        VirtualProductType::ShouldJump => {
            let mut jump_flags: Vec<u8> = vec![0; len];
            let mut not_next_noop: Vec<u8> = vec![0; len];

            jump_flags
                .par_iter_mut()
                .zip(not_next_noop.par_iter_mut())
                .enumerate()
                .for_each(|(i, (jump, not_noop))| {
                    *jump = trace[i].instruction().circuit_flags()[CircuitFlags::Jump] as u8;

                    let is_next_noop = if i + 1 < len {
                        trace[i + 1].instruction().circuit_flags()[CircuitFlags::IsNoop] as u8
                    } else {
                        1 // Last cycle, treat as if next is NoOp
                    };
                    *not_noop = 1 - is_next_noop;
                });

            (jump_flags.into(), not_next_noop.into())
        }
    }
}

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
