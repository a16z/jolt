//! R1CS input catalog and per-cycle typed views
//!
//! - Canonical enumeration and ordering of all virtual inputs consumed by the
//!   Spartan outer sumcheck: `JoltR1CSInputs` and `ALL_R1CS_INPUTS`. Provides
//!   indices and conversions to `VirtualPolynomial` and to `OpeningId`.
//! - Materialized, single-cycle views sourced from the execution trace:
//!   - `R1CSCycleInputs`: full row used by uniform R1CS and shift constraints;
//!     built via `from_trace` with exact integer semantics.
//!   - `ProductCycleInputs`: minimal tuple for the product virtualization sumcheck;
//!     the de-duplicated factor list is `PRODUCT_UNIQUE_FACTOR_VIRTUALS`.
//!
//! Maintainers: keep the enum order, `ALL_R1CS_INPUTS`, and `to_index` in sync.
//! Changes here affect `r1cs::constraints` (row shapes) and `r1cs::evaluation`
//! (typed evaluators and claim computation).

use crate::poly::opening_proof::{OpeningId, SumcheckId};
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::instruction::{
    CircuitFlags, Flags, InstructionFlags, LookupQuery, NUM_CIRCUIT_FLAGS,
};
use crate::zkvm::witness::VirtualPolynomial;

use crate::field::JoltField;
use ark_ff::biginteger::{S128, S64};
use common::constants::XLEN;
use std::fmt::Debug;
use tracer::instruction::Cycle;

use strum::IntoEnumIterator;

/// Inputs to the Spartan outer sumcheck. All is virtual, each produce a claim for later stages
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JoltR1CSInputs {
    PC,                    // (bytecode raf)
    UnexpandedPC,          // (bytecode rv)
    Imm,                   // (bytecode rv)
    RamAddress,            // (RAM raf)
    Rs1Value,              // (registers rv)
    Rs2Value,              // (registers rv)
    RdWriteValue,          // (registers wv)
    RamReadValue,          // (RAM rv)
    RamWriteValue,         // (RAM wv)
    LeftInstructionInput,  // (instruction input)
    RightInstructionInput, // (instruction input)
    LeftLookupOperand,     // (instruction raf)
    RightLookupOperand,    // (instruction raf)
    Product,               // (product virtualization)
    WriteLookupOutputToRD, // (product virtualization)
    WritePCtoRD,           // (product virtualization)
    ShouldBranch,          // (product virtualization)
    NextUnexpandedPC,      // (shift sumcheck)
    NextPC,                // (shift sumcheck)
    NextIsVirtual,         // (shift sumcheck)
    NextIsFirstInSequence, // (shift sumcheck)
    LookupOutput,          // (instruction rv)
    ShouldJump,            // (product virtualization)
    OpFlags(CircuitFlags),
}

pub const NUM_R1CS_INPUTS: usize = ALL_R1CS_INPUTS.len();
/// This const serves to define a canonical ordering over inputs (and thus indices
/// for each input). This is needed for sumcheck.
pub const ALL_R1CS_INPUTS: [JoltR1CSInputs; 36] = [
    JoltR1CSInputs::LeftInstructionInput,
    JoltR1CSInputs::RightInstructionInput,
    JoltR1CSInputs::Product,
    JoltR1CSInputs::WriteLookupOutputToRD,
    JoltR1CSInputs::WritePCtoRD,
    JoltR1CSInputs::ShouldBranch,
    JoltR1CSInputs::PC,
    JoltR1CSInputs::UnexpandedPC,
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
    JoltR1CSInputs::NextIsVirtual,
    JoltR1CSInputs::NextIsFirstInSequence,
    JoltR1CSInputs::LookupOutput,
    JoltR1CSInputs::ShouldJump,
    JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands),
    JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands),
    JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands),
    JoltR1CSInputs::OpFlags(CircuitFlags::Load),
    JoltR1CSInputs::OpFlags(CircuitFlags::Store),
    JoltR1CSInputs::OpFlags(CircuitFlags::Jump),
    JoltR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToRD),
    JoltR1CSInputs::OpFlags(CircuitFlags::VirtualInstruction),
    JoltR1CSInputs::OpFlags(CircuitFlags::Assert),
    JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC),
    JoltR1CSInputs::OpFlags(CircuitFlags::Advice),
    JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed),
    JoltR1CSInputs::OpFlags(CircuitFlags::IsFirstInSequence),
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
            JoltR1CSInputs::Imm => 8,
            JoltR1CSInputs::RamAddress => 9,
            JoltR1CSInputs::Rs1Value => 10,
            JoltR1CSInputs::Rs2Value => 11,
            JoltR1CSInputs::RdWriteValue => 12,
            JoltR1CSInputs::RamReadValue => 13,
            JoltR1CSInputs::RamWriteValue => 14,
            JoltR1CSInputs::LeftLookupOperand => 15,
            JoltR1CSInputs::RightLookupOperand => 16,
            JoltR1CSInputs::NextUnexpandedPC => 17,
            JoltR1CSInputs::NextPC => 18,
            JoltR1CSInputs::NextIsVirtual => 19,
            JoltR1CSInputs::NextIsFirstInSequence => 20,
            JoltR1CSInputs::LookupOutput => 21,
            JoltR1CSInputs::ShouldJump => 22,
            JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands) => 23,
            JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands) => 24,
            JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands) => 25,
            JoltR1CSInputs::OpFlags(CircuitFlags::Load) => 26,
            JoltR1CSInputs::OpFlags(CircuitFlags::Store) => 27,
            JoltR1CSInputs::OpFlags(CircuitFlags::Jump) => 28,
            JoltR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToRD) => 29,
            JoltR1CSInputs::OpFlags(CircuitFlags::VirtualInstruction) => 30,
            JoltR1CSInputs::OpFlags(CircuitFlags::Assert) => 31,
            JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC) => 32,
            JoltR1CSInputs::OpFlags(CircuitFlags::Advice) => 33,
            JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed) => 34,
            JoltR1CSInputs::OpFlags(CircuitFlags::IsFirstInSequence) => 35,
        }
    }
}

impl From<&JoltR1CSInputs> for VirtualPolynomial {
    fn from(input: &JoltR1CSInputs) -> Self {
        match input {
            JoltR1CSInputs::PC => VirtualPolynomial::PC,
            JoltR1CSInputs::UnexpandedPC => VirtualPolynomial::UnexpandedPC,
            JoltR1CSInputs::Imm => VirtualPolynomial::Imm,
            JoltR1CSInputs::RamAddress => VirtualPolynomial::RamAddress,
            JoltR1CSInputs::Rs1Value => VirtualPolynomial::Rs1Value,
            JoltR1CSInputs::Rs2Value => VirtualPolynomial::Rs2Value,
            JoltR1CSInputs::RdWriteValue => VirtualPolynomial::RdWriteValue,
            JoltR1CSInputs::RamReadValue => VirtualPolynomial::RamReadValue,
            JoltR1CSInputs::RamWriteValue => VirtualPolynomial::RamWriteValue,
            JoltR1CSInputs::LeftLookupOperand => VirtualPolynomial::LeftLookupOperand,
            JoltR1CSInputs::RightLookupOperand => VirtualPolynomial::RightLookupOperand,
            JoltR1CSInputs::Product => VirtualPolynomial::Product,
            JoltR1CSInputs::NextUnexpandedPC => VirtualPolynomial::NextUnexpandedPC,
            JoltR1CSInputs::NextPC => VirtualPolynomial::NextPC,
            JoltR1CSInputs::LookupOutput => VirtualPolynomial::LookupOutput,
            JoltR1CSInputs::ShouldJump => VirtualPolynomial::ShouldJump,
            JoltR1CSInputs::ShouldBranch => VirtualPolynomial::ShouldBranch,
            JoltR1CSInputs::WritePCtoRD => VirtualPolynomial::WritePCtoRD,
            JoltR1CSInputs::WriteLookupOutputToRD => VirtualPolynomial::WriteLookupOutputToRD,
            JoltR1CSInputs::OpFlags(flag) => VirtualPolynomial::OpFlags(*flag),
            JoltR1CSInputs::LeftInstructionInput => VirtualPolynomial::LeftInstructionInput,
            JoltR1CSInputs::RightInstructionInput => VirtualPolynomial::RightInstructionInput,
            JoltR1CSInputs::NextIsVirtual => VirtualPolynomial::NextIsVirtual,
            JoltR1CSInputs::NextIsFirstInSequence => VirtualPolynomial::NextIsFirstInSequence,
        }
    }
}

impl From<&JoltR1CSInputs> for OpeningId {
    fn from(input: &JoltR1CSInputs) -> Self {
        let poly = VirtualPolynomial::from(input);
        OpeningId::Virtual(poly, SumcheckId::SpartanOuter)
    }
}

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
    /// Derived: `Branch && (LookupOutput == 1)`.
    pub should_branch: bool,

    /// `IsRdNotZero` && ` `WriteLookupOutputToRD`
    pub write_lookup_output_to_rd_addr: bool,
    /// `IsRdNotZero` && `Jump`
    pub write_pc_to_rd_addr: bool,

    /// `VirtualInstruction` flag for the next cycle (false for last cycle).
    pub next_is_virtual: bool,
    /// `FirstInSequence` flag for the next cycle (false for last cycle).
    pub next_is_first_in_sequence: bool,
}

impl R1CSCycleInputs {
    /// Build directly from the execution trace and preprocessing,
    /// mirroring the optimized semantics used in `compute_claimed_r1cs_input_evals`.
    pub fn from_trace<F>(
        bytecode_preprocessing: &BytecodePreprocessing,
        trace: &[Cycle],
        t: usize,
    ) -> Self
    where
        F: JoltField,
    {
        let len = trace.len();
        let cycle = &trace[t];
        let instr = cycle.instruction();
        let flags_view = instr.circuit_flags();
        let instruction_flags = instr.instruction_flags();
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
        let pc = bytecode_preprocessing.get_pc(cycle) as u64;
        let next_pc = if let Some(nc) = next_cycle {
            bytecode_preprocessing.get_pc(nc) as u64
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
            nc.instruction().instruction_flags()[InstructionFlags::IsNoop]
        } else {
            false // There is no next cycle, so cannot be a noop
        };
        let should_jump = flags_view[CircuitFlags::Jump] && !next_is_noop;
        let should_branch = instruction_flags[InstructionFlags::Branch] && (lookup_output == 1);

        // Write-to-Rd selectors (masked by flags)
        let write_lookup_output_to_rd_addr = flags_view[CircuitFlags::WriteLookupOutputToRD]
            && instruction_flags[InstructionFlags::IsRdNotZero];
        let write_pc_to_rd_addr =
            flags_view[CircuitFlags::Jump] && instruction_flags[InstructionFlags::IsRdNotZero];

        let (next_is_virtual, next_is_first_in_sequence) = if let Some(nc) = next_cycle {
            let flags = nc.instruction().circuit_flags();
            (
                flags[CircuitFlags::VirtualInstruction],
                flags[CircuitFlags::IsFirstInSequence],
            )
        } else {
            (false, false)
        };

        Self {
            left_input,
            right_input,
            product,
            left_lookup,
            right_lookup,
            lookup_output,
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
            next_is_virtual,
            next_is_first_in_sequence,
        }
    }
}

/// Canonical, de-duplicated list of product-virtual factor polynomials used by
/// the Product Virtualization stage.
/// Order:
/// 0: LeftInstructionInput
/// 1: RightInstructionInput
/// 2: InstructionFlags(IsRdNotZero)
/// 3: OpFlags(WriteLookupOutputToRD)
/// 4: OpFlags(Jump)
/// 5: LookupOutput
/// 6: InstructionFlags(Branch)
/// 7: NextIsNoop
pub const PRODUCT_UNIQUE_FACTOR_VIRTUALS: [VirtualPolynomial; 8] = [
    VirtualPolynomial::LeftInstructionInput,
    VirtualPolynomial::RightInstructionInput,
    VirtualPolynomial::InstructionFlags(InstructionFlags::IsRdNotZero),
    VirtualPolynomial::OpFlags(CircuitFlags::WriteLookupOutputToRD),
    VirtualPolynomial::OpFlags(CircuitFlags::Jump),
    VirtualPolynomial::LookupOutput,
    VirtualPolynomial::InstructionFlags(InstructionFlags::Branch),
    VirtualPolynomial::NextIsNoop,
];

/// Minimal, unified view for the Product-virtualization round: the 5 product pairs
/// (left, right) materialized from the trace for a single cycle.
/// Total size is small; we keep primitive representations that match witness generation.
#[derive(Clone, Debug)]
pub struct ProductCycleInputs {
    // 16-byte aligned
    /// Instruction: LeftInstructionInput × RightInstructionInput (right input as i128)
    pub instruction_right_input: i128,

    // 8-byte aligned
    pub instruction_left_input: u64,
    /// ShouldBranch: LookupOutput × Branch_flag (left side)
    pub should_branch_lookup_output: u64,

    // 1-byte fields
    /// WriteLookupOutputToRD right flag (boolean)
    pub write_lookup_output_to_rd_flag: bool,
    /// Jump flag used by both WritePCtoRD (right) and ShouldJump (left)
    pub jump_flag: bool,
    /// ShouldBranch right flag (boolean)
    pub should_branch_flag: bool,
    /// ShouldJump right flag (1 - NextIsNoop)
    pub not_next_noop: bool,
    /// IsRdNotZero instruction flag (boolean)
    pub is_rd_not_zero: bool,
}

impl ProductCycleInputs {
    /// Build from trace and preprocessing, mirroring the semantics used by
    /// product-virtualization witness generation.
    pub fn from_trace<F>(trace: &[Cycle], t: usize) -> Self
    where
        F: JoltField,
    {
        let len = trace.len();
        let cycle = &trace[t];
        let instr = cycle.instruction();
        let flags_view = instr.circuit_flags();
        let instruction_flags = instr.instruction_flags();

        // Instruction inputs
        let (left_input, right_input) = LookupQuery::<XLEN>::to_instruction_inputs(cycle);

        // Lookup output
        let lookup_output = LookupQuery::<XLEN>::to_lookup_output(cycle);

        // Jump and Branch flags
        let jump_flag = flags_view[CircuitFlags::Jump];
        let branch_flag = instruction_flags[InstructionFlags::Branch];

        // Next-is-noop and its complement (1 - NextIsNoop)
        let not_next_noop = {
            if t + 1 < len {
                !trace[t + 1].instruction().instruction_flags()[InstructionFlags::IsNoop]
            } else {
                // Needs final not_next_noop to be false for the shift sumcheck
                // (since EqPlusOne does not do overflow)
                false
            }
        };

        let is_rd_not_zero = instruction_flags[InstructionFlags::IsRdNotZero];

        // WriteLookupOutputToRD flag
        let write_lookup_output_to_rd_flag = flags_view[CircuitFlags::WriteLookupOutputToRD];

        Self {
            instruction_left_input: left_input,
            instruction_right_input: right_input,
            write_lookup_output_to_rd_flag,
            should_branch_lookup_output: lookup_output,
            should_branch_flag: branch_flag,
            jump_flag,
            not_next_noop,
            is_rd_not_zero,
        }
    }
}

// TODO(Quang): move the shift sumcheck inputs to here as well

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
                (JoltR1CSInputs::NextIsVirtual, JoltR1CSInputs::NextIsVirtual) => true,
                (JoltR1CSInputs::NextIsFirstInSequence, JoltR1CSInputs::NextIsFirstInSequence) => {
                    true
                }
                (JoltR1CSInputs::LookupOutput, JoltR1CSInputs::LookupOutput) => true,
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
                (CircuitFlags::AddOperands, CircuitFlags::AddOperands)
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
                    | (
                        CircuitFlags::WriteLookupOutputToRD,
                        CircuitFlags::WriteLookupOutputToRD
                    )
                    | (
                        CircuitFlags::VirtualInstruction,
                        CircuitFlags::VirtualInstruction
                    )
                    | (CircuitFlags::Assert, CircuitFlags::Assert)
                    | (
                        CircuitFlags::DoNotUpdateUnexpandedPC,
                        CircuitFlags::DoNotUpdateUnexpandedPC
                    )
                    | (CircuitFlags::Advice, CircuitFlags::Advice)
                    | (CircuitFlags::IsCompressed, CircuitFlags::IsCompressed)
                    | (
                        CircuitFlags::IsFirstInSequence,
                        CircuitFlags::IsFirstInSequence
                    )
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
