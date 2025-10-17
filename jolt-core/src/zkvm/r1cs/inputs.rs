use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::opening_proof::{OpeningId, SumcheckId};
use crate::utils::accumulation::{
    acc5u_add_field, acc5u_fmadd_u64, acc5u_new, acc5u_reduce, acc6s_fmadd_i128, acc6s_new,
    acc6s_reduce, acc6u_fmadd_u64, acc6u_new, acc6u_reduce, acc7s_fmadd_s128, acc7s_new,
    acc7s_reduce, acc7u_fmadd_u128, acc7u_new, acc7u_reduce,
};
#[cfg(test)]
use crate::utils::small_scalar::SmallScalar;
use crate::zkvm::instruction::{
    CircuitFlags, Flags, InstructionFlags, LookupQuery, NUM_CIRCUIT_FLAGS,
};
use crate::zkvm::spartan::product::VirtualProductType;
use crate::zkvm::witness::VirtualPolynomial;
use crate::zkvm::JoltSharedPreprocessing;

use crate::field::JoltField;
use ark_ff::biginteger::{S128, S64};
use ark_std::Zero;
use common::constants::XLEN;
use rayon::prelude::*;
use std::fmt::Debug;
use tracer::instruction::Cycle;

use strum::IntoEnumIterator;

/// The inputs to R1CS constraints. Everything is virtual.
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
    LeftInstructionInput,  // (spartan instruction input)
    RightInstructionInput, // (spartan instruction input)
    LeftLookupOperand,     // (instruction lookup raf)
    RightLookupOperand,    // (instruction lookup raf)
    Product,               // (spartan product virtual)
    WriteLookupOutputToRD, // (spartan product virtual)
    WritePCtoRD,           // (spartan product virtual)
    ShouldBranch,          // (spartan product virtual)
    NextUnexpandedPC,      // (spartan pc shift)
    NextPC,                // (spartan pc shift)
    LookupOutput,          // (instruction rv)
    ShouldJump,            // (spartan product virtual)
    OpFlags(CircuitFlags),
}

const NUM_R1CS_INPUTS: usize = ALL_R1CS_INPUTS.len();
/// This const serves to define a canonical ordering over inputs (and thus indices
/// for each input). This is needed for sumcheck.
pub const ALL_R1CS_INPUTS: [JoltR1CSInputs; 33] = [
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
    JoltR1CSInputs::LookupOutput,
    JoltR1CSInputs::ShouldJump,
    JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands),
    JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands),
    JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands),
    JoltR1CSInputs::OpFlags(CircuitFlags::Load),
    JoltR1CSInputs::OpFlags(CircuitFlags::Store),
    JoltR1CSInputs::OpFlags(CircuitFlags::Jump),
    JoltR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToRD),
    JoltR1CSInputs::OpFlags(CircuitFlags::InlineSequenceInstruction),
    JoltR1CSInputs::OpFlags(CircuitFlags::Assert),
    JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC),
    JoltR1CSInputs::OpFlags(CircuitFlags::Advice),
    JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed),
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
            JoltR1CSInputs::LookupOutput => 19,
            JoltR1CSInputs::ShouldJump => 20,
            JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands) => 21,
            JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands) => 22,
            JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands) => 23,
            JoltR1CSInputs::OpFlags(CircuitFlags::Load) => 24,
            JoltR1CSInputs::OpFlags(CircuitFlags::Store) => 25,
            JoltR1CSInputs::OpFlags(CircuitFlags::Jump) => 26,
            JoltR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToRD) => 27,
            JoltR1CSInputs::OpFlags(CircuitFlags::InlineSequenceInstruction) => 28,
            JoltR1CSInputs::OpFlags(CircuitFlags::Assert) => 29,
            JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC) => 30,
            JoltR1CSInputs::OpFlags(CircuitFlags::Advice) => 31,
            JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed) => 32,
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
    // 16-byte aligned first to minimize padding
    /// Right lookup operand (u128) for the instruction lookup query.
    /// Full-width integer encoding used by add/sub/mul/advice cases.
    pub right_lookup: u128,

    // Next largest-by-size, 8-byte alignment group
    /// Signed-magnitude `S128` product consistent with the `Product` witness.
    /// Computed from `left_input` × `right_input` using the same truncation semantics as the witness.
    pub product: S128,
    /// Right instruction input as signed-magnitude `S64`.
    /// Typically `Imm` or `Rs2Value` with exact integer semantics.
    pub right_input: S64,
    /// Immediate operand as signed-magnitude `S64`.
    pub imm: S64,

    // 8-byte scalars
    /// Left instruction input as a u64 bit-pattern.
    /// Typically `Rs1Value` or the current `UnexpandedPC`, depending on `CircuitFlags`.
    pub left_input: u64,
    /// Left lookup operand (u64) for the instruction lookup query.
    /// Matches `LeftLookupOperand` virtual polynomial semantics.
    pub left_lookup: u64,
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

    // 1-byte fields last to pack tightly
    /// Destination register index (Rd).
    pub rd_addr: u8,

    /// Rd index if `WriteLookupOutputToRD`, else 0 (u8 domain used as selector).
    pub write_lookup_output_to_rd_addr: u8,
    /// Rd index if `Jump`, else 0 (u8 domain used as selector).
    pub write_pc_to_rd_addr: u8,

    /// Per-instruction circuit flags indexed by `CircuitFlags`.
    pub flags: [bool; NUM_CIRCUIT_FLAGS],
    /// `IsNoop` flag for the next cycle (false for last cycle).
    pub next_is_noop: bool,

    /// Derived: `Jump && !NextIsNoop`.
    pub should_jump: bool,
    /// Derived: `Branch && (LookupOutput == 1)`.
    pub should_branch: bool,
}

impl R1CSCycleInputs {
    /// Build directly from the execution trace and preprocessing,
    /// mirroring the optimized semantics used in `compute_claimed_r1cs_input_evals`.
    pub fn from_trace<F>(preprocessing: &JoltSharedPreprocessing, trace: &[Cycle], t: usize) -> Self
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
            nc.instruction().instruction_flags()[InstructionFlags::IsNoop]
        } else {
            false
        };
        let should_jump = flags_view[CircuitFlags::Jump] && !next_is_noop;
        let should_branch = instruction_flags[InstructionFlags::Branch] && (lookup_output == 1);

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
            JoltR1CSInputs::ShouldBranch => F::from_bool(self.should_branch),
            JoltR1CSInputs::PC => self.pc.to_field(),
            JoltR1CSInputs::UnexpandedPC => self.unexpanded_pc.to_field(),
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
            JoltR1CSInputs::ShouldJump => F::from_bool(self.should_jump),
            JoltR1CSInputs::OpFlags(flag) => F::from_bool(self.flags[flag]),
        }
    }
}

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
    /// WriteLookupOutputToRD: RdWa × WriteLookupOutputToRD_flag (left side)
    pub write_lookup_output_to_rd_rd_addr: u8,
    /// WritePCtoRD: RdWa × Jump_flag (left side)
    pub write_pc_to_rd_rd_addr: u8,

    /// WriteLookupOutputToRD right flag (boolean)
    pub write_lookup_output_to_rd_flag: bool,
    /// WritePCtoRD right flag (boolean)
    pub write_pc_to_rd_flag: bool,
    /// ShouldBranch right flag (boolean)
    pub should_branch_flag: bool,
    /// ShouldJump left flag (Jump)
    pub should_jump_flag: bool,
    /// ShouldJump right flag (1 - NextIsNoop)
    pub not_next_noop: bool,
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

        // Rd address
        let rd_addr = cycle.rd_write().0;

        // Jump and Branch flags
        let jump_flag = flags_view[CircuitFlags::Jump];
        let branch_flag = instruction_flags[InstructionFlags::Branch];

        // Next-is-noop and its complement (1 - NextIsNoop)
        let not_next_noop = {
            let is_next_noop = if t + 1 < len {
                trace[t + 1].instruction().instruction_flags()[InstructionFlags::IsNoop]
            } else {
                true // Treat last cycle as if next is NoOp
            };
            !is_next_noop
        };

        // WriteLookupOutputToRD flag
        let write_lookup_output_to_rd_flag = flags_view[CircuitFlags::WriteLookupOutputToRD];

        Self {
            instruction_left_input: left_input,
            instruction_right_input: right_input,
            write_lookup_output_to_rd_rd_addr: rd_addr,
            write_lookup_output_to_rd_flag,
            write_pc_to_rd_rd_addr: rd_addr,
            write_pc_to_rd_flag: jump_flag,
            should_branch_lookup_output: lookup_output,
            should_branch_flag: branch_flag,
            should_jump_flag: jump_flag,
            not_next_noop,
        }
    }
}
/// Compute `z(r_cycle) = Σ_t eq(r_cycle, t) * P_i(t)` for all inputs i, without
/// materializing P_i. Returns `[P_0(r_cycle), P_1(r_cycle), ...]` in input order.
#[tracing::instrument(skip_all)]
pub fn compute_claimed_r1cs_input_evals<F: JoltField>(
    preprocessing: &JoltSharedPreprocessing,
    trace: &[Cycle],
    r_cycle: &[F::Challenge],
) -> Vec<F> {
    // Double-sum with delayed reduction using typed accumulators per input
    let m = r_cycle.len() / 2;
    let (r2, r1) = r_cycle.split_at(m);
    let (eq_one, eq_two) = rayon::join(|| EqPolynomial::evals(r2), || EqPolynomial::evals(r1));

    let total_unr: [F::Unreduced<9>; NUM_R1CS_INPUTS] = (0..eq_one.len())
        .into_par_iter()
        .map(|x1| {
            let eq1_val = eq_one[x1];
            // Accumulators ordered by category
            let mut acc_left_input = acc6u_new::<F>();
            let mut acc_right_input = acc6s_new::<F>();
            let mut acc_product = acc7s_new::<F>();
            let mut acc_wl_left = acc5u_new::<F>();
            let mut acc_wp_left = acc5u_new::<F>();
            let mut acc_pc = acc6u_new::<F>();
            let mut acc_unexpanded_pc = acc6u_new::<F>();
            let mut acc_imm = acc6s_new::<F>();
            let mut acc_ram_address = acc6u_new::<F>();
            let mut acc_rs1_value = acc6u_new::<F>();
            let mut acc_rs2_value = acc6u_new::<F>();
            let mut acc_rd_write_value = acc6u_new::<F>();
            let mut acc_ram_read_value = acc6u_new::<F>();
            let mut acc_ram_write_value = acc6u_new::<F>();
            let mut acc_left_lookup_operand = acc6u_new::<F>();
            let mut acc_right_lookup_operand = acc7u_new::<F>();
            let mut acc_next_unexpanded_pc = acc6u_new::<F>();
            let mut acc_next_pc = acc6u_new::<F>();
            let mut acc_lookup_output = acc6u_new::<F>();
            let mut acc_sj_flag = acc5u_new::<F>();
            let mut acc_sb_right = acc5u_new::<F>();
            let mut acc_flags: Vec<F::Unreduced<5>> =
                (0..NUM_CIRCUIT_FLAGS).map(|_| acc5u_new::<F>()).collect();

            for x2 in 0..eq_two.len() {
                let e_in = eq_two[x2];
                let idx = x1 * eq_two.len() + x2;
                let row = R1CSCycleInputs::from_trace::<F>(preprocessing, trace, idx);

                acc6u_fmadd_u64(&mut acc_left_input, &e_in, row.left_input);
                acc6s_fmadd_i128(&mut acc_right_input, &e_in, row.right_input.to_i128());
                acc7s_fmadd_s128(&mut acc_product, &e_in, row.product);

                acc5u_fmadd_u64(
                    &mut acc_wl_left,
                    &e_in,
                    row.write_lookup_output_to_rd_addr as u64,
                );
                acc5u_fmadd_u64(&mut acc_wp_left, &e_in, row.write_pc_to_rd_addr as u64);
                if row.should_branch {
                    acc5u_add_field(&mut acc_sb_right, &e_in);
                }

                acc6u_fmadd_u64(&mut acc_pc, &e_in, row.pc);
                acc6u_fmadd_u64(&mut acc_unexpanded_pc, &e_in, row.unexpanded_pc);
                acc6s_fmadd_i128(&mut acc_imm, &e_in, row.imm.to_i128());
                acc6u_fmadd_u64(&mut acc_ram_address, &e_in, row.ram_addr);
                acc6u_fmadd_u64(&mut acc_rs1_value, &e_in, row.rs1_read_value);
                acc6u_fmadd_u64(&mut acc_rs2_value, &e_in, row.rs2_read_value);
                acc6u_fmadd_u64(&mut acc_rd_write_value, &e_in, row.rd_write_value);
                acc6u_fmadd_u64(&mut acc_ram_read_value, &e_in, row.ram_read_value);
                acc6u_fmadd_u64(&mut acc_ram_write_value, &e_in, row.ram_write_value);
                acc6u_fmadd_u64(&mut acc_left_lookup_operand, &e_in, row.left_lookup);
                acc7u_fmadd_u128(&mut acc_right_lookup_operand, &e_in, row.right_lookup);
                acc6u_fmadd_u64(&mut acc_next_unexpanded_pc, &e_in, row.next_unexpanded_pc);
                acc6u_fmadd_u64(&mut acc_next_pc, &e_in, row.next_pc);
                acc6u_fmadd_u64(&mut acc_lookup_output, &e_in, row.lookup_output);
                if row.should_jump {
                    acc5u_add_field(&mut acc_sj_flag, &e_in);
                }
                for flag in CircuitFlags::iter() {
                    if row.flags[flag] {
                        let idx = flag as usize;
                        acc5u_add_field(&mut acc_flags[idx], &e_in);
                    }
                }
            }

            // Reduce per-input and apply E_out unreduced
            let mut out_unr: [F::Unreduced<9>; NUM_R1CS_INPUTS] =
                [F::Unreduced::<9>::zero(); NUM_R1CS_INPUTS];
            out_unr[JoltR1CSInputs::LeftInstructionInput.to_index()] =
                eq1_val.mul_unreduced::<9>(acc6u_reduce::<F>(&acc_left_input));
            out_unr[JoltR1CSInputs::RightInstructionInput.to_index()] =
                eq1_val.mul_unreduced::<9>(acc6s_reduce::<F>(&acc_right_input));
            out_unr[JoltR1CSInputs::Product.to_index()] =
                eq1_val.mul_unreduced::<9>(acc7s_reduce::<F>(&acc_product));
            out_unr[JoltR1CSInputs::WriteLookupOutputToRD.to_index()] =
                eq1_val.mul_unreduced::<9>(acc5u_reduce::<F>(&acc_wl_left));
            out_unr[JoltR1CSInputs::WritePCtoRD.to_index()] =
                eq1_val.mul_unreduced::<9>(acc5u_reduce::<F>(&acc_wp_left));
            out_unr[JoltR1CSInputs::ShouldBranch.to_index()] =
                eq1_val.mul_unreduced::<9>(acc5u_reduce::<F>(&acc_sb_right));
            out_unr[JoltR1CSInputs::PC.to_index()] =
                eq1_val.mul_unreduced::<9>(acc6u_reduce::<F>(&acc_pc));
            out_unr[JoltR1CSInputs::UnexpandedPC.to_index()] =
                eq1_val.mul_unreduced::<9>(acc6u_reduce::<F>(&acc_unexpanded_pc));
            out_unr[JoltR1CSInputs::Imm.to_index()] =
                eq1_val.mul_unreduced::<9>(acc6s_reduce::<F>(&acc_imm));
            out_unr[JoltR1CSInputs::RamAddress.to_index()] =
                eq1_val.mul_unreduced::<9>(acc6u_reduce::<F>(&acc_ram_address));
            out_unr[JoltR1CSInputs::Rs1Value.to_index()] =
                eq1_val.mul_unreduced::<9>(acc6u_reduce::<F>(&acc_rs1_value));
            out_unr[JoltR1CSInputs::Rs2Value.to_index()] =
                eq1_val.mul_unreduced::<9>(acc6u_reduce::<F>(&acc_rs2_value));
            out_unr[JoltR1CSInputs::RdWriteValue.to_index()] =
                eq1_val.mul_unreduced::<9>(acc6u_reduce::<F>(&acc_rd_write_value));
            out_unr[JoltR1CSInputs::RamReadValue.to_index()] =
                eq1_val.mul_unreduced::<9>(acc6u_reduce::<F>(&acc_ram_read_value));
            out_unr[JoltR1CSInputs::RamWriteValue.to_index()] =
                eq1_val.mul_unreduced::<9>(acc6u_reduce::<F>(&acc_ram_write_value));
            out_unr[JoltR1CSInputs::LeftLookupOperand.to_index()] =
                eq1_val.mul_unreduced::<9>(acc6u_reduce::<F>(&acc_left_lookup_operand));
            out_unr[JoltR1CSInputs::RightLookupOperand.to_index()] =
                eq1_val.mul_unreduced::<9>(acc7u_reduce::<F>(&acc_right_lookup_operand));
            out_unr[JoltR1CSInputs::NextUnexpandedPC.to_index()] =
                eq1_val.mul_unreduced::<9>(acc6u_reduce::<F>(&acc_next_unexpanded_pc));
            out_unr[JoltR1CSInputs::NextPC.to_index()] =
                eq1_val.mul_unreduced::<9>(acc6u_reduce::<F>(&acc_next_pc));
            out_unr[JoltR1CSInputs::LookupOutput.to_index()] =
                eq1_val.mul_unreduced::<9>(acc6u_reduce::<F>(&acc_lookup_output));
            out_unr[JoltR1CSInputs::ShouldJump.to_index()] =
                eq1_val.mul_unreduced::<9>(acc5u_reduce::<F>(&acc_sj_flag));
            for flag in CircuitFlags::iter() {
                let idx = JoltR1CSInputs::OpFlags(flag).to_index();
                let f_idx = flag as usize;
                out_unr[idx] = eq1_val.mul_unreduced::<9>(acc5u_reduce::<F>(&acc_flags[f_idx]));
            }
            out_unr
        })
        .reduce(
            || [F::Unreduced::<9>::zero(); NUM_R1CS_INPUTS],
            |mut acc, item| {
                for i in 0..NUM_R1CS_INPUTS {
                    acc[i] += item[i];
                }
                acc
            },
        );

    // Final Montgomery reduce per entry
    (0..NUM_R1CS_INPUTS)
        .map(|i| F::from_montgomery_reduce::<9>(total_unr[i]))
        .collect()
}

/// Compute z(r_cycle) for the 10 product-virtualization inputs (left/right for each product).
/// Order of outputs:
/// 0: Instruction.left, 1: Instruction.right,
/// 2: WriteLookupOutputToRD.left(rd_addr), 3: WriteLookupOutputToRD.right(flag),
/// 4: WritePCtoRD.left(rd_addr), 5: WritePCtoRD.right(jump_flag),
/// 6: ShouldBranch.left(lookup_output), 7: ShouldBranch.right(branch_flag),
/// 8: ShouldJump.left(jump_flag), 9: 1 - ShouldJump.right (NextIsNoop)
#[tracing::instrument(skip_all)]
pub fn compute_claimed_product_virtual_evals<F: JoltField>(
    trace: &[Cycle],
    r_cycle: &[F::Challenge],
) -> [F; 10] {
    let m = r_cycle.len() / 2;
    let (r2, r1) = r_cycle.split_at(m);
    let (eq_one, eq_two) = rayon::join(|| EqPolynomial::evals(r2), || EqPolynomial::evals(r1));

    let eq_two_len = eq_two.len();

    let totals_unr: [F::Unreduced<9>; 10] = (0..eq_one.len())
        .into_par_iter()
        .map(|x1| {
            let eq1_val = eq_one[x1];

            // Instruction: (left u64, right i128)
            let mut acc_inst_left = acc6u_new::<F>();
            let mut acc_inst_right = acc6s_new::<F>();
            // WriteLookupOutputToRD: (left u8, right bool)
            let mut acc_wl_left = acc5u_new::<F>();
            let mut acc_wl_right = acc5u_new::<F>();
            // WritePCtoRD: (left u8, right bool)
            let mut acc_wp_left = acc5u_new::<F>();
            let mut acc_wp_right = acc5u_new::<F>();
            // ShouldBranch: (left u64, right bool)
            let mut acc_sb_left = acc6u_new::<F>();
            let mut acc_sb_right = acc5u_new::<F>();
            // ShouldJump: (left bool, right bool)
            let mut acc_sj_left = acc5u_new::<F>();
            let mut acc_sj_right = acc5u_new::<F>();

            for x2 in 0..eq_two_len {
                let e_in = eq_two[x2];
                let idx = x1 * eq_two_len + x2;
                let row = ProductCycleInputs::from_trace::<F>(trace, idx);

                // Instruction: (u64, i128)
                acc6u_fmadd_u64(&mut acc_inst_left, &e_in, row.instruction_left_input);
                acc6s_fmadd_i128(&mut acc_inst_right, &e_in, row.instruction_right_input);

                // WriteLookupOutputToRD: (u8, bool)
                acc5u_fmadd_u64(
                    &mut acc_wl_left,
                    &e_in,
                    row.write_lookup_output_to_rd_rd_addr as u64,
                );
                if row.write_lookup_output_to_rd_flag {
                    acc5u_add_field(&mut acc_wl_right, &e_in);
                }

                // WritePCtoRD: (u8, bool)
                acc5u_fmadd_u64(&mut acc_wp_left, &e_in, row.write_pc_to_rd_rd_addr as u64);
                if row.write_pc_to_rd_flag {
                    acc5u_add_field(&mut acc_wp_right, &e_in);
                }

                // ShouldBranch: (u64, bool)
                acc6u_fmadd_u64(&mut acc_sb_left, &e_in, row.should_branch_lookup_output);
                if row.should_branch_flag {
                    acc5u_add_field(&mut acc_sb_right, &e_in);
                }

                // ShouldJump: (bool, bool)
                if row.should_jump_flag {
                    acc5u_add_field(&mut acc_sj_left, &e_in);
                }
                if !row.not_next_noop {
                    acc5u_add_field(&mut acc_sj_right, &e_in);
                }
            }

            // Reduce per-output and apply E_out unreduced
            let mut out_unr = [F::Unreduced::<9>::zero(); 10];
            out_unr[0] = eq1_val.mul_unreduced::<9>(acc6u_reduce::<F>(&acc_inst_left));
            out_unr[1] = eq1_val.mul_unreduced::<9>(acc6s_reduce::<F>(&acc_inst_right));
            out_unr[2] = eq1_val.mul_unreduced::<9>(acc5u_reduce::<F>(&acc_wl_left));
            out_unr[3] = eq1_val.mul_unreduced::<9>(acc5u_reduce::<F>(&acc_wl_right));
            out_unr[4] = eq1_val.mul_unreduced::<9>(acc5u_reduce::<F>(&acc_wp_left));
            out_unr[5] = eq1_val.mul_unreduced::<9>(acc5u_reduce::<F>(&acc_wp_right));
            out_unr[6] = eq1_val.mul_unreduced::<9>(acc6u_reduce::<F>(&acc_sb_left));
            out_unr[7] = eq1_val.mul_unreduced::<9>(acc5u_reduce::<F>(&acc_sb_right));
            out_unr[8] = eq1_val.mul_unreduced::<9>(acc5u_reduce::<F>(&acc_sj_left));
            out_unr[9] = eq1_val.mul_unreduced::<9>(acc5u_reduce::<F>(&acc_sj_right));

            out_unr
        })
        .reduce(
            || [F::Unreduced::<9>::zero(); 10],
            |mut acc, item| {
                for i in 0..10 {
                    acc[i] += item[i];
                }
                acc
            },
        );

    // Final reduce
    let mut out: [F; 10] = [F::zero(); 10];
    for i in 0..10 {
        out[i] = F::from_montgomery_reduce::<9>(totals_unr[i]);
    }
    out
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
            *n = cycle.instruction().instruction_flags()[InstructionFlags::IsNoop] as u8;
        });

    (unexpanded_pc.into(), pc.into(), is_noop.into())
}

// TODO(markosg04): we could unify this with the `generate_witness_batch` to avoid a second iteration over T
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
        VirtualProductType::Instruction => {
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
                    *flag = cycle.instruction().instruction_flags()[InstructionFlags::Branch] as u8;
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
                        trace[i + 1].instruction().instruction_flags()[InstructionFlags::IsNoop]
                            as u8
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
                        CircuitFlags::InlineSequenceInstruction,
                        CircuitFlags::InlineSequenceInstruction
                    )
                    | (CircuitFlags::Assert, CircuitFlags::Assert)
                    | (
                        CircuitFlags::DoNotUpdateUnexpandedPC,
                        CircuitFlags::DoNotUpdateUnexpandedPC
                    )
                    | (CircuitFlags::Advice, CircuitFlags::Advice)
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
    fn r1cs_cycle_inputs_size_and_alignment() {
        use core::mem::{align_of, size_of};
        assert_eq!(align_of::<R1CSCycleInputs>(), 16, "unexpected alignment");
        assert_eq!(size_of::<R1CSCycleInputs>(), 208, "unexpected size");
    }
}
