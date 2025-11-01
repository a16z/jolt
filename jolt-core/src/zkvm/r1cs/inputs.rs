use crate::poly::eq_poly::EqPolynomial;
use crate::poly::opening_proof::{OpeningId, SumcheckId};
use crate::utils::accumulation::{Acc5U, Acc6S, Acc6U, Acc7S, Acc7U};
use crate::zkvm::instruction::{
    CircuitFlags, Flags, InstructionFlags, LookupQuery, NUM_CIRCUIT_FLAGS,
};
use crate::zkvm::witness::VirtualPolynomial;
use crate::zkvm::JoltSharedPreprocessing;

use crate::field::{BarrettReduce, FMAdd, JoltField};
use ark_ff::biginteger::{S128, S64};
use ark_std::Zero;
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

const NUM_R1CS_INPUTS: usize = ALL_R1CS_INPUTS.len();
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

/// Compute `z(r_cycle) = Σ_t eq(r_cycle, t) * P_i(t)` for all inputs i, without
/// materializing P_i. Returns `[P_0(r_cycle), P_1(r_cycle), ...]` in input order.
#[tracing::instrument(skip_all)]
pub fn compute_claimed_r1cs_input_evals<F: JoltField>(
    preprocessing: &JoltSharedPreprocessing,
    trace: &[Cycle],
    r_cycle: &[F::Challenge],
) -> [F; NUM_R1CS_INPUTS] {
    let m = r_cycle.len() / 2;
    let (r2, r1) = r_cycle.split_at(m);
    let (eq_one, eq_two) = rayon::join(|| EqPolynomial::evals(r2), || EqPolynomial::evals(r1));

    (0..eq_one.len())
        .into_par_iter()
        .map(|x1| {
            let eq1_val = eq_one[x1];

            // (DON'T DELETE) Accumulators for each input
            // If bool or u8 => 5 limbs unsigned
            // If u64 => 6 limbs unsigned
            // If i128 => 6 limbs signed
            // If S128 => 7 limbs signed
            let mut acc_left_input: Acc6U<F> = Acc6U::default();
            let mut acc_right_input: Acc6S<F> = Acc6S::default();
            let mut acc_product: Acc7S<F> = Acc7S::default();
            let mut acc_wl_left: Acc5U<F> = Acc5U::default();
            let mut acc_wp_left: Acc5U<F> = Acc5U::default();
            let mut acc_sb_right: Acc5U<F> = Acc5U::default();
            let mut acc_pc: Acc6U<F> = Acc6U::default();
            let mut acc_unexpanded_pc: Acc6U<F> = Acc6U::default();
            let mut acc_imm: Acc6S<F> = Acc6S::default();
            let mut acc_ram_address: Acc6U<F> = Acc6U::default();
            let mut acc_rs1_value: Acc6U<F> = Acc6U::default();
            let mut acc_rs2_value: Acc6U<F> = Acc6U::default();
            let mut acc_rd_write_value: Acc6U<F> = Acc6U::default();
            let mut acc_ram_read_value: Acc6U<F> = Acc6U::default();
            let mut acc_ram_write_value: Acc6U<F> = Acc6U::default();
            let mut acc_left_lookup_operand: Acc6U<F> = Acc6U::default();
            let mut acc_right_lookup_operand: Acc7U<F> = Acc7U::default();
            let mut acc_next_unexpanded_pc: Acc6U<F> = Acc6U::default();
            let mut acc_next_pc: Acc6U<F> = Acc6U::default();
            let mut acc_lookup_output: Acc6U<F> = Acc6U::default();
            let mut acc_sj_flag: Acc5U<F> = Acc5U::default();
            let mut acc_next_is_virtual: Acc5U<F> = Acc5U::default();
            let mut acc_next_is_first_in_sequence: Acc5U<F> = Acc5U::default();
            let mut acc_flags: Vec<Acc5U<F>> =
                (0..NUM_CIRCUIT_FLAGS).map(|_| Acc5U::default()).collect();

            let eq_two_len = eq_two.len();
            for x2 in 0..eq_two_len {
                let e_in = eq_two[x2];
                let idx = x1 * eq_two_len + x2;
                let row = R1CSCycleInputs::from_trace::<F>(preprocessing, trace, idx);

                acc_left_input.fmadd(&e_in, &row.left_input);
                acc_right_input.fmadd(&e_in, &row.right_input.to_i128());
                acc_product.fmadd(&e_in, &row.product);

                acc_wl_left.fmadd(&e_in, &(row.write_lookup_output_to_rd_addr as u64));
                acc_wp_left.fmadd(&e_in, &(row.write_pc_to_rd_addr as u64));
                acc_sb_right.fmadd(&e_in, &row.should_branch);

                acc_pc.fmadd(&e_in, &row.pc);
                acc_unexpanded_pc.fmadd(&e_in, &row.unexpanded_pc);
                acc_imm.fmadd(&e_in, &row.imm.to_i128());
                acc_ram_address.fmadd(&e_in, &row.ram_addr);
                acc_rs1_value.fmadd(&e_in, &row.rs1_read_value);
                acc_rs2_value.fmadd(&e_in, &row.rs2_read_value);
                acc_rd_write_value.fmadd(&e_in, &row.rd_write_value);
                acc_ram_read_value.fmadd(&e_in, &row.ram_read_value);
                acc_ram_write_value.fmadd(&e_in, &row.ram_write_value);
                acc_left_lookup_operand.fmadd(&e_in, &row.left_lookup);
                acc_right_lookup_operand.fmadd(&e_in, &row.right_lookup);
                acc_next_unexpanded_pc.fmadd(&e_in, &row.next_unexpanded_pc);
                acc_next_pc.fmadd(&e_in, &row.next_pc);
                acc_lookup_output.fmadd(&e_in, &row.lookup_output);
                acc_sj_flag.fmadd(&e_in, &row.should_jump);
                acc_next_is_virtual.fmadd(&e_in, &row.next_is_virtual);
                acc_next_is_first_in_sequence.fmadd(&e_in, &row.next_is_first_in_sequence);
                for flag in CircuitFlags::iter() {
                    acc_flags[flag as usize].fmadd(&e_in, &row.flags[flag as usize]);
                }
            }

            let mut out_unr: [F::Unreduced<9>; NUM_R1CS_INPUTS] =
                [F::Unreduced::<9>::zero(); NUM_R1CS_INPUTS];
            out_unr[JoltR1CSInputs::LeftInstructionInput.to_index()] =
                eq1_val.mul_unreduced::<9>(acc_left_input.barrett_reduce());
            out_unr[JoltR1CSInputs::RightInstructionInput.to_index()] =
                eq1_val.mul_unreduced::<9>(acc_right_input.barrett_reduce());
            out_unr[JoltR1CSInputs::Product.to_index()] =
                eq1_val.mul_unreduced::<9>(acc_product.barrett_reduce());
            out_unr[JoltR1CSInputs::WriteLookupOutputToRD.to_index()] =
                eq1_val.mul_unreduced::<9>(acc_wl_left.barrett_reduce());
            out_unr[JoltR1CSInputs::WritePCtoRD.to_index()] =
                eq1_val.mul_unreduced::<9>(acc_wp_left.barrett_reduce());
            out_unr[JoltR1CSInputs::ShouldBranch.to_index()] =
                eq1_val.mul_unreduced::<9>(acc_sb_right.barrett_reduce());
            out_unr[JoltR1CSInputs::PC.to_index()] =
                eq1_val.mul_unreduced::<9>(acc_pc.barrett_reduce());
            out_unr[JoltR1CSInputs::UnexpandedPC.to_index()] =
                eq1_val.mul_unreduced::<9>(acc_unexpanded_pc.barrett_reduce());
            out_unr[JoltR1CSInputs::Imm.to_index()] =
                eq1_val.mul_unreduced::<9>(acc_imm.barrett_reduce());
            out_unr[JoltR1CSInputs::RamAddress.to_index()] =
                eq1_val.mul_unreduced::<9>(acc_ram_address.barrett_reduce());
            out_unr[JoltR1CSInputs::Rs1Value.to_index()] =
                eq1_val.mul_unreduced::<9>(acc_rs1_value.barrett_reduce());
            out_unr[JoltR1CSInputs::Rs2Value.to_index()] =
                eq1_val.mul_unreduced::<9>(acc_rs2_value.barrett_reduce());
            out_unr[JoltR1CSInputs::RdWriteValue.to_index()] =
                eq1_val.mul_unreduced::<9>(acc_rd_write_value.barrett_reduce());
            out_unr[JoltR1CSInputs::RamReadValue.to_index()] =
                eq1_val.mul_unreduced::<9>(acc_ram_read_value.barrett_reduce());
            out_unr[JoltR1CSInputs::RamWriteValue.to_index()] =
                eq1_val.mul_unreduced::<9>(acc_ram_write_value.barrett_reduce());
            out_unr[JoltR1CSInputs::LeftLookupOperand.to_index()] =
                eq1_val.mul_unreduced::<9>(acc_left_lookup_operand.barrett_reduce());
            out_unr[JoltR1CSInputs::RightLookupOperand.to_index()] =
                eq1_val.mul_unreduced::<9>(acc_right_lookup_operand.barrett_reduce());
            out_unr[JoltR1CSInputs::NextUnexpandedPC.to_index()] =
                eq1_val.mul_unreduced::<9>(acc_next_unexpanded_pc.barrett_reduce());
            out_unr[JoltR1CSInputs::NextPC.to_index()] =
                eq1_val.mul_unreduced::<9>(acc_next_pc.barrett_reduce());
            out_unr[JoltR1CSInputs::LookupOutput.to_index()] =
                eq1_val.mul_unreduced::<9>(acc_lookup_output.barrett_reduce());
            out_unr[JoltR1CSInputs::ShouldJump.to_index()] =
                eq1_val.mul_unreduced::<9>(acc_sj_flag.barrett_reduce());
            out_unr[JoltR1CSInputs::NextIsVirtual.to_index()] =
                eq1_val.mul_unreduced::<9>(acc_next_is_virtual.barrett_reduce());
            out_unr[JoltR1CSInputs::NextIsFirstInSequence.to_index()] =
                eq1_val.mul_unreduced::<9>(acc_next_is_first_in_sequence.barrett_reduce());
            for flag in CircuitFlags::iter() {
                let idx = JoltR1CSInputs::OpFlags(flag).to_index();
                let f_idx = flag as usize;
                out_unr[idx] = eq1_val.mul_unreduced::<9>(acc_flags[f_idx].barrett_reduce());
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
        )
        .map(|unr| F::from_montgomery_reduce::<9>(unr))
}

/// Canonical, de-duplicated list of product-virtual factor polynomials used by
/// the Product Virtualization stage (in stable order).
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

    /// Compute both fused left and right factors at r and return as a pair (left, right).
    ///
    /// Expected order of weights:
    /// [Instruction, WriteLookupOutputToRD, WritePCtoRD, ShouldBranch, ShouldJump]
    pub fn compute_left_right_at_r<F: JoltField>(&self, weights_at_r0: &[F]) -> (F, F) {
        // Left: u64/u8/bool
        let mut left_acc: Acc6U<F> = Acc6U::default();
        left_acc.fmadd(&weights_at_r0[0], &self.instruction_left_input);
        left_acc.fmadd(&weights_at_r0[1], &self.is_rd_not_zero);
        left_acc.fmadd(&weights_at_r0[2], &self.is_rd_not_zero);
        left_acc.fmadd(&weights_at_r0[3], &self.should_branch_lookup_output);
        left_acc.fmadd(&weights_at_r0[4], &self.jump_flag);

        // Right: i128/bool
        let mut right_acc: Acc6S<F> = Acc6S::default();
        right_acc.fmadd(&weights_at_r0[0], &self.instruction_right_input);
        right_acc.fmadd(&weights_at_r0[1], &self.write_lookup_output_to_rd_flag);
        right_acc.fmadd(&weights_at_r0[2], &self.jump_flag);
        right_acc.fmadd(&weights_at_r0[3], &self.should_branch_flag);
        right_acc.fmadd(&weights_at_r0[4], &self.not_next_noop);

        (left_acc.barrett_reduce(), right_acc.barrett_reduce())
    }
}

/// Compute z(r_cycle) for the 8 de-duplicated factor polynomials used by Product Virtualization.
/// Order of outputs matches PRODUCT_UNIQUE_FACTOR_VIRTUALS:
/// 0: LeftInstructionInput (u64)
/// 1: RightInstructionInput (i128)
/// 2: IsRdNotZero (bool)
/// 3: OpFlags(WriteLookupOutputToRD) (bool)
/// 4: OpFlags(Jump) (bool)
/// 5: LookupOutput (u64)
/// 6: InstructionFlags(Branch) (bool)
/// 7: NextIsNoop (bool)
#[tracing::instrument(skip_all)]
pub fn compute_claimed_product_factor_evals<F: JoltField>(
    trace: &[Cycle],
    r_cycle: &[F::Challenge],
) -> [F; 8] {
    let m = r_cycle.len() / 2;
    let (r2, r1) = r_cycle.split_at(m);
    let (eq_one, eq_two) = rayon::join(|| EqPolynomial::evals(r2), || EqPolynomial::evals(r1));

    let eq_two_len = eq_two.len();

    let totals_unr: [F::Unreduced<9>; 8] = (0..eq_one.len())
        .into_par_iter()
        .map(|x1| {
            let eq1_val = eq_one[x1];

            // Accumulators for 8 outputs
            let mut acc_left_u64: Acc6U<F> = Acc6U::default();
            let mut acc_right_i128: Acc6S<F> = Acc6S::default();
            let mut acc_rd_zero_flag: Acc5U<F> = Acc5U::default();
            let mut acc_wl_flag: Acc5U<F> = Acc5U::default();
            let mut acc_jump_flag: Acc5U<F> = Acc5U::default();
            let mut acc_lookup_output: Acc6U<F> = Acc6U::default();
            let mut acc_branch_flag: Acc5U<F> = Acc5U::default();
            let mut acc_next_is_noop: Acc5U<F> = Acc5U::default();

            for x2 in 0..eq_two_len {
                let e_in = eq_two[x2];
                let idx = x1 * eq_two_len + x2;
                let row = ProductCycleInputs::from_trace::<F>(trace, idx);

                // 0: LeftInstructionInput (u64)
                acc_left_u64.fmadd(&e_in, &row.instruction_left_input);
                // 1: RightInstructionInput (i128)
                acc_right_i128.fmadd(&e_in, &row.instruction_right_input);
                // 2: IsRdZero (bool)
                acc_rd_zero_flag.fmadd(&e_in, &(row.is_rd_not_zero));
                // 3: OpFlags(WriteLookupOutputToRD) (bool)
                acc_wl_flag.fmadd(&e_in, &row.write_lookup_output_to_rd_flag);
                // 4: OpFlags(Jump) (bool)
                acc_jump_flag.fmadd(&e_in, &row.jump_flag);
                // 5: LookupOutput (u64)
                acc_lookup_output.fmadd(&e_in, &row.should_branch_lookup_output);
                // 6: InstructionFlags(Branch) (bool)
                acc_branch_flag.fmadd(&e_in, &row.should_branch_flag);
                // 7: NextIsNoop (bool) = !not_next_noop
                acc_next_is_noop.fmadd(&e_in, &(!row.not_next_noop));
            }

            let mut out_unr = [F::Unreduced::<9>::zero(); 8];
            out_unr[0] = eq1_val.mul_unreduced::<9>(acc_left_u64.barrett_reduce());
            out_unr[1] = eq1_val.mul_unreduced::<9>(acc_right_i128.barrett_reduce());
            out_unr[2] = eq1_val.mul_unreduced::<9>(acc_rd_zero_flag.barrett_reduce());
            out_unr[3] = eq1_val.mul_unreduced::<9>(acc_wl_flag.barrett_reduce());
            out_unr[4] = eq1_val.mul_unreduced::<9>(acc_jump_flag.barrett_reduce());
            out_unr[5] = eq1_val.mul_unreduced::<9>(acc_lookup_output.barrett_reduce());
            out_unr[6] = eq1_val.mul_unreduced::<9>(acc_branch_flag.barrett_reduce());
            out_unr[7] = eq1_val.mul_unreduced::<9>(acc_next_is_noop.barrett_reduce());
            out_unr
        })
        .reduce(
            || [F::Unreduced::<9>::zero(); 8],
            |mut acc, item| {
                for i in 0..8 {
                    acc[i] += item[i];
                }
                acc
            },
        );

    core::array::from_fn(|i| F::from_montgomery_reduce::<9>(totals_unr[i]))
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
