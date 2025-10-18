//! Compile-time constant R1CS constraints
//!
//! This module provides a static, compile-time representation of R1CS constraints
//! to replace the dynamic constraint building in the prover's hot path.
//!
//! ## Adding a new constraint
//!
//! To add a new R1CS constraint:
//! 1. Add a new variant to `ConstraintName` enum
//! 2. Add the constraint to `UNIFORM_R1CS` array using appropriate macro
//! 3. Add custom evaluators for grouped Az/Bz
//!
//! ## Removing a constraint
//!
//! To remove an R1CS constraint:
//! 1. Remove the constraint from `UNIFORM_R1CS` array
//! 2. Remove the corresponding variant from `ConstraintName` enum
//! 3. Remove any custom evaluator from `eval_az_bz_by_name`
//!
//! ## Custom evaluators
//!
//! Custom evaluators provide optimized Az/Bz evaluation
//! using `SmallScalar` types to avoid field conversions. They should:
//! - Use appropriate `I8OrI96` variants (prefer `Bool` or `I8` for flags/small sums)
//! - Use appropriate `S160` variants (prefer `U64AndSign` for u64-u64 diffs)
//! - Only use `U128AndSign` when 128-bit arithmetic is inherently required

use super::inputs::{JoltR1CSInputs, R1CSCycleInputs};
use crate::field::JoltField;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::zkvm::instruction::CircuitFlags;
use ark_ff::biginteger::S160;
use strum::EnumCount;
use strum_macros::{EnumCount, EnumIter};

pub use super::ops::{Term, LC};

/// A single R1CS constraint row
#[derive(Clone, Copy, Debug)]
pub struct Constraint {
    pub a: LC,
    pub b: LC,
    pub c: LC,
}

impl Constraint {
    pub const fn new(a: LC, b: LC, c: LC) -> Self {
        Self { a, b, c }
    }

    /// Evaluate this constraint at a specific row in the witness polynomials
    /// Returns (a_eval, b_eval, c_eval) tuple
    #[inline]
    pub fn evaluate_row<F: JoltField>(
        &self,
        flattened_polynomials: &[MultilinearPolynomial<F>],
        row: usize,
    ) -> (F, F, F) {
        let a_eval = self.a.evaluate_row(flattened_polynomials, row);
        let b_eval = self.b.evaluate_row(flattened_polynomials, row);
        let c_eval = self.c.evaluate_row(flattened_polynomials, row);
        (a_eval, b_eval, c_eval)
    }
}

impl LC {
    /// Evaluate this LC given the inputs for a R1CS cycle, using field semantics, only for testing
    #[cfg(test)]
    pub fn evaluate_row_with<F: JoltField>(&self, inputs: &R1CSCycleInputs) -> F {
        let mut result = F::zero();
        self.for_each_term(|input_index, coeff| {
            result += crate::utils::small_scalar::SmallScalar::field_mul(
                &coeff,
                inputs.to_field::<F>(JoltR1CSInputs::from_index(input_index)),
            );
        });
        if let Some(c) = self.const_term() {
            result += crate::utils::small_scalar::SmallScalar::to_field::<F>(c);
        }
        result
    }
}

/// Creates: condition * (left - right) == 0
pub const fn constraint_eq_conditional_lc(condition: LC, left: LC, right: LC) -> Constraint {
    Constraint::new(
        condition,
        match left.checked_sub(right) {
            Some(b) => b,
            None => LC::zero(),
        },
        LC::zero(),
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, EnumCount, EnumIter)]
pub enum ConstraintName {
    RamAddrEqRs1PlusImmIfLoadStore,
    RamAddrEqZeroIfNotLoadStore,
    RamReadEqRamWriteIfLoad,
    RamReadEqRdWriteIfLoad,
    Rs2EqRamWriteIfStore,
    LeftLookupZeroUnlessAddSubMul,
    LeftLookupEqLeftInputOtherwise,
    RightLookupAdd,
    RightLookupSub,
    RightLookupEqProductIfMul,
    RightLookupEqRightInputOtherwise,
    AssertLookupOne,
    RdWriteEqLookupIfWriteLookupToRd,
    RdWriteEqPCPlusConstIfWritePCtoRD,
    NextUnexpPCEqLookupIfShouldJump,
    NextUnexpPCEqPCPlusImmIfShouldBranch,
    NextUnexpPCUpdateOtherwise,
    NextPCEqPCPlusOneIfInline,
    MustStartSequenceFromBeginning,
}

#[derive(Clone, Copy, Debug)]
pub struct NamedConstraint {
    pub name: ConstraintName,
    pub cons: Constraint,
}

/// Creates: left * right == result
pub const fn constraint_prod_lc(left: LC, right: LC, result: LC) -> Constraint {
    Constraint::new(left, right, result)
}

/// Creates: condition * (true_val - false_val) == (result - false_val)
pub const fn constraint_if_else_lc(
    condition: LC,
    true_val: LC,
    false_val: LC,
    result: LC,
) -> Constraint {
    Constraint::new(
        condition,
        match true_val.checked_sub(false_val) {
            Some(b) => b,
            None => LC::zero(),
        },
        match result.checked_sub(false_val) {
            Some(c) => c,
            None => LC::zero(),
        },
    )
}

/// r1cs_eq_conditional!: verbose, condition-first equality constraint
///
/// Usage: `r1cs_eq_conditional!(name: ConstraintName::Foo, if { COND } => { LEFT } == { RIGHT });`
#[macro_export]
macro_rules! r1cs_eq_conditional {
    (name: $nm:expr, if { $($cond:tt)* } => ( $($left:tt)* ) == ( $($right:tt)* ) ) => {{
        $crate::zkvm::r1cs::constraints::NamedConstraint {
            name: $nm,
            cons: $crate::zkvm::r1cs::constraints::constraint_eq_conditional_lc(
                $crate::lc!($($cond)*),
                $crate::lc!($($left)*),
                $crate::lc!($($right)*),
            ),
        }
    }};
}

/// Number of uniform R1CS constraints
pub const NUM_R1CS_CONSTRAINTS: usize = ConstraintName::COUNT;

/// Static table of all R1CS uniform constraints.
pub static UNIFORM_R1CS: [NamedConstraint; NUM_R1CS_CONSTRAINTS] = [
    // if Load || Store {
    //     assert!(RamAddress == Rs1Value + Imm)
    // } else {
    //     assert!(RamAddress == 0)
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::RamAddrEqRs1PlusImmIfLoadStore,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::Load) } + { JoltR1CSInputs::OpFlags(CircuitFlags::Store) } }
        => ( { JoltR1CSInputs::RamAddress } ) == ( { JoltR1CSInputs::Rs1Value } + { JoltR1CSInputs::Imm } )
    ),
    r1cs_eq_conditional!(
        name: ConstraintName::RamAddrEqZeroIfNotLoadStore,
        if { { 1i128 } - { JoltR1CSInputs::OpFlags(CircuitFlags::Load) } - { JoltR1CSInputs::OpFlags(CircuitFlags::Store) } }
        => ( { JoltR1CSInputs::RamAddress } ) == ( { 0i128 } )
    ),
    // if Load {
    //     assert!(RamReadValue == RamWriteValue)
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::RamReadEqRamWriteIfLoad,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::Load) } }
        => ( { JoltR1CSInputs::RamReadValue } ) == ( { JoltR1CSInputs::RamWriteValue } )
    ),
    // if Load {
    //     assert!(RamReadValue == RdWriteValue)
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::RamReadEqRdWriteIfLoad,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::Load) } }
        => ( { JoltR1CSInputs::RamReadValue } ) == ( { JoltR1CSInputs::RdWriteValue } )
    ),
    // if Store {
    //     assert!(Rs2Value == RamWriteValue)
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::Rs2EqRamWriteIfStore,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::Store) } }
        => ( { JoltR1CSInputs::Rs2Value } ) == ( { JoltR1CSInputs::RamWriteValue } )
    ),
    // if AddOperands || SubtractOperands || MultiplyOperands {
    //     // Lookup query is just RightLookupOperand
    //     assert!(LeftLookupOperand == 0)
    // } else {
    //     assert!(LeftLookupOperand == LeftInstructionInput)
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::LeftLookupZeroUnlessAddSubMul,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands) } + { JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands) } + { JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands) } }
        => ( { JoltR1CSInputs::LeftLookupOperand } ) == ( { 0i128 } )
    ),
    r1cs_eq_conditional!(
        name: ConstraintName::LeftLookupEqLeftInputOtherwise,
        if { { 1i128 } - { JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands) } - { JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands) } - { JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands) } }
        => ( { JoltR1CSInputs::LeftLookupOperand } ) == ( { JoltR1CSInputs::LeftInstructionInput } )
    ),
    // If AddOperands {
    //     assert!(RightLookupOperand == LeftInstructionInput + RightInstructionInput)
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::RightLookupAdd,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands) } }
        => ( { JoltR1CSInputs::RightLookupOperand } ) == ( { JoltR1CSInputs::LeftInstructionInput } + { JoltR1CSInputs::RightInstructionInput } )
    ),
    // If SubtractOperands {
    //     assert!(RightLookupOperand == LeftInstructionInput - RightInstructionInput)
    // }
    // Converts from unsigned to twos-complement representation
    r1cs_eq_conditional!(
        name: ConstraintName::RightLookupSub,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands) } }
        => ( { JoltR1CSInputs::RightLookupOperand } ) == ( { JoltR1CSInputs::LeftInstructionInput } - { JoltR1CSInputs::RightInstructionInput } + { 0x10000000000000000i128 } )
    ),
    r1cs_eq_conditional!(
        name: ConstraintName::RightLookupEqProductIfMul,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands) } }
        => ( { JoltR1CSInputs::RightLookupOperand } ) == ( { JoltR1CSInputs::Product } )
    ),
    // if !(AddOperands || SubtractOperands || MultiplyOperands || Advice) {
    //     assert!(RightLookupOperand == RightInstructionInput)
    // }
    // Arbitrary untrusted advice goes in right lookup operand
    r1cs_eq_conditional!(
        name: ConstraintName::RightLookupEqRightInputOtherwise,
        if { { 1i128 } - { JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands) } - { JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands) } - { JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands) } - { JoltR1CSInputs::OpFlags(CircuitFlags::Advice) } }
        => ( { JoltR1CSInputs::RightLookupOperand } ) == ( { JoltR1CSInputs::RightInstructionInput } )
    ),
    // if Assert {
    //     assert!(LookupOutput == 1)
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::AssertLookupOne,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::Assert) } }
        => ( { JoltR1CSInputs::LookupOutput } ) == ( { 1i128 } )
    ),
    // if Rd != 0 && WriteLookupOutputToRD {
    //     assert!(RdWriteValue == LookupOutput)
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::RdWriteEqLookupIfWriteLookupToRd,
        if { { JoltR1CSInputs::WriteLookupOutputToRD } }
        => ( { JoltR1CSInputs::RdWriteValue } ) == ( { JoltR1CSInputs::LookupOutput } )
    ),
    // if Rd != 0 && Jump {
    //     if !isCompressed {
    //          assert!(RdWriteValue == UnexpandedPC + 4)
    //     } else {
    //          assert!(RdWriteValue == UnexpandedPC + 2)
    //     }
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::RdWriteEqPCPlusConstIfWritePCtoRD,
        if { { JoltR1CSInputs::WritePCtoRD } }
        => ( { JoltR1CSInputs::RdWriteValue } ) == ( { JoltR1CSInputs::UnexpandedPC } + { 4i128 } - { 2 * JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed) } )
    ),
    // if Jump && !NextIsNoop {
    //     assert!(NextUnexpandedPC == LookupOutput)
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::NextUnexpPCEqLookupIfShouldJump,
        if { { JoltR1CSInputs::ShouldJump } }
        => ( { JoltR1CSInputs::NextUnexpandedPC } ) == ( { JoltR1CSInputs::LookupOutput } )
    ),
    // if Branch && LookupOutput {
    //     assert!(NextUnexpandedPC == UnexpandedPC + Imm)
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::NextUnexpPCEqPCPlusImmIfShouldBranch,
        if { { JoltR1CSInputs::ShouldBranch } }
        => ( { JoltR1CSInputs::NextUnexpandedPC } ) == ( { JoltR1CSInputs::UnexpandedPC } + { JoltR1CSInputs::Imm } )
    ),
    // if !(ShouldBranch || Jump) {
    //     if DoNotUpdatePC {
    //         assert!(NextUnexpandedPC == UnexpandedPC)
    //     } else if isCompressed {
    //         assert!(NextUnexpandedPC == UnexpandedPC + 2)
    //     } else {
    //         assert!(NextUnexpandedPC == UnexpandedPC + 4)
    //     }
    // }
    // Note that ShouldBranch and Jump instructions are mutually exclusive
    // And that DoNotUpdatePC and isCompressed are mutually exclusive
    r1cs_eq_conditional!(
        name: ConstraintName::NextUnexpPCUpdateOtherwise,
        if { { 1i128 } - { JoltR1CSInputs::ShouldBranch } - { JoltR1CSInputs::OpFlags(CircuitFlags::Jump) } }
        => ( { JoltR1CSInputs::NextUnexpandedPC } )
           == ( { JoltR1CSInputs::UnexpandedPC } + { 4i128 }
                - { 4 * JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC) }
                - { 2 * JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed) } )
    ),
    // if Inline {
    //     assert!(NextPC == PC + 1)
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::NextPCEqPCPlusOneIfInline,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::VirtualInstruction) } }
        => ( { JoltR1CSInputs::NextPC } ) == ( { JoltR1CSInputs::PC } + { 1i128 } )
    ),
    // if NextIsVirtual && !NextIsFirstInSequence {
    //     assert!(DoNotUpdateUnexpandedPC == 1)
    // }
    r1cs_eq_conditional!(
        name: ConstraintName::MustStartSequenceFromBeginning,
        if { { JoltR1CSInputs::NextIsVirtual } - { JoltR1CSInputs::NextIsFirstInSequence } }
        => ( { JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC) } ) == ( { 1 } )
    ),
];

// =============================================================================
// Univariate skip constants and grouped views
// =============================================================================

/// Degree of univariate skip, defined to be `(NUM_R1CS_CONSTRAINTS - 1) / 2`
pub const UNIVARIATE_SKIP_DEGREE: usize = (NUM_R1CS_CONSTRAINTS - 1) / 2;

/// Domain size of univariate skip, defined to be `UNIVARIATE_SKIP_DEGREE + 1`.
pub const UNIVARIATE_SKIP_DOMAIN_SIZE: usize = UNIVARIATE_SKIP_DEGREE + 1;

/// Extended domain size of univariate skip, defined to be `2 * UNIVARIATE_SKIP_DEGREE + 1`.
pub const UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE: usize = 2 * UNIVARIATE_SKIP_DEGREE + 1;

/// Number of coefficients in the first-round polynomial, defined to be `3 * UNIVARIATE_SKIP_DEGREE + 1`.
pub const FIRST_ROUND_POLY_NUM_COEFFS: usize = 3 * UNIVARIATE_SKIP_DEGREE + 1;

/// Number of remaining R1CS constraints in the second group, defined to be
/// `NUM_R1CS_CONSTRAINTS - UNIVARIATE_SKIP_DOMAIN_SIZE`.
pub const NUM_REMAINING_R1CS_CONSTRAINTS: usize =
    NUM_R1CS_CONSTRAINTS - UNIVARIATE_SKIP_DOMAIN_SIZE;

/// Order-preserving, compile-time filter over `UNIFORM_R1CS` by constraint names.
const fn contains_name<const N: usize>(names: &[ConstraintName; N], name: ConstraintName) -> bool {
    let mut i = 0;
    while i < N {
        if names[i] as u32 == name as u32 {
            return true;
        }
        i += 1;
    }
    false
}

/// Select constraints from `UNIFORM_R1CS` whose names appear in `names`, preserving order.
pub const fn filter_uniform_r1cs<const N: usize>(
    names: &[ConstraintName; N],
) -> [NamedConstraint; N] {
    let dummy = NamedConstraint {
        name: ConstraintName::RamReadEqRamWriteIfLoad,
        cons: Constraint::new(LC::zero(), LC::zero(), LC::zero()),
    };
    let mut out: [NamedConstraint; N] = [dummy; N];

    let mut o = 0;
    let mut i = 0;
    while i < NUM_R1CS_CONSTRAINTS {
        let cand = UNIFORM_R1CS[i];
        if contains_name(names, cand.name) {
            out[o] = cand;
            o += 1;
            if o == N {
                break;
            }
        }
        i += 1;
    }

    if o != N {
        panic!("filter_uniform_r1cs: not all requested constraints were found in UNIFORM_R1CS");
    }
    out
}

/// Compute the complement of `UNIFORM_R1CS_FIRST_GROUP_NAMES` within `UNIFORM_R1CS`.
const fn complement_first_group_names() -> [ConstraintName; NUM_REMAINING_R1CS_CONSTRAINTS] {
    let mut out: [ConstraintName; NUM_REMAINING_R1CS_CONSTRAINTS] =
        [ConstraintName::RamReadEqRamWriteIfLoad; NUM_REMAINING_R1CS_CONSTRAINTS];
    let mut o = 0;
    let mut i = 0;
    while i < NUM_R1CS_CONSTRAINTS {
        let cand = UNIFORM_R1CS[i].name;
        if !contains_name(&UNIFORM_R1CS_FIRST_GROUP_NAMES, cand) {
            out[o] = cand;
            o += 1;
            if o == NUM_REMAINING_R1CS_CONSTRAINTS {
                break;
            }
        }
        i += 1;
    }

    if o != NUM_REMAINING_R1CS_CONSTRAINTS {
        panic!("complement_first_group_names: expected full complement");
    }
    out
}

/// First group: 10 boolean-guarded eq constraints, where Bz is around 64 bits
pub const UNIFORM_R1CS_FIRST_GROUP_NAMES: [ConstraintName; UNIVARIATE_SKIP_DOMAIN_SIZE] = [
    ConstraintName::RamAddrEqZeroIfNotLoadStore,
    ConstraintName::RamReadEqRamWriteIfLoad,
    ConstraintName::RamReadEqRdWriteIfLoad,
    ConstraintName::Rs2EqRamWriteIfStore,
    ConstraintName::LeftLookupZeroUnlessAddSubMul,
    ConstraintName::LeftLookupEqLeftInputOtherwise,
    ConstraintName::AssertLookupOne,
    ConstraintName::NextUnexpPCEqLookupIfShouldJump,
    ConstraintName::NextPCEqPCPlusOneIfInline,
    ConstraintName::MustStartSequenceFromBeginning,
];

/// Second group: complement of first within UNIFORM_R1CS
/// Here, Az may be u8, and Bz may be around 128 bits
pub const UNIFORM_R1CS_SECOND_GROUP_NAMES: [ConstraintName; NUM_REMAINING_R1CS_CONSTRAINTS] =
    complement_first_group_names();

/// First group: 10 boolean-guarded eq constraints, where Bz is around 64 bits
pub static UNIFORM_R1CS_FIRST_GROUP: [NamedConstraint; UNIVARIATE_SKIP_DOMAIN_SIZE] =
    filter_uniform_r1cs(&UNIFORM_R1CS_FIRST_GROUP_NAMES);

/// Second group: complement of first within UNIFORM_R1CS, where Az may be u8 and Bz may be around 128 bits
pub static UNIFORM_R1CS_SECOND_GROUP: [NamedConstraint; NUM_REMAINING_R1CS_CONSTRAINTS] =
    filter_uniform_r1cs(&UNIFORM_R1CS_SECOND_GROUP_NAMES);

/// Evaluate Az for the first group
pub fn eval_az_first_group(row: &R1CSCycleInputs) -> [bool; UNIVARIATE_SKIP_DOMAIN_SIZE] {
    let flags = &row.flags;
    let ld = flags[CircuitFlags::Load];
    let st = flags[CircuitFlags::Store];
    let add = flags[CircuitFlags::AddOperands];
    let sub = flags[CircuitFlags::SubtractOperands];
    let mul = flags[CircuitFlags::MultiplyOperands];
    let assert_flag = flags[CircuitFlags::Assert];
    let inline_seq = flags[CircuitFlags::VirtualInstruction];

    [
        !(ld || st),
        ld,
        ld,
        st,
        add || sub || mul,
        !(add || sub || mul),
        assert_flag,
        row.should_jump,
        inline_seq,
        row.next_is_virtual && !row.next_is_first_in_sequence,
    ]
}

/// Evaluate Bz for the first group
pub fn eval_bz_first_group(row: &R1CSCycleInputs) -> [i128; UNIVARIATE_SKIP_DOMAIN_SIZE] {
    let left_lookup = row.left_lookup as i128;
    let left_input = row.left_input as i128;
    let ram_read = row.ram_read_value as i128;
    let ram_write = row.ram_write_value as i128;
    let rd_write = row.rd_write_value as i128;
    let rs2 = row.rs2_read_value as i128;
    let ram_addr = row.ram_addr as i128;
    let lookup_out = row.lookup_output as i128;
    let next_unexp_pc = row.next_unexpanded_pc as i128;
    let pc = row.pc as i128;
    let next_pc = row.next_pc as i128;

    [
        // RamAddrEqZeroIfNotLoadStore: RamAddress - 0
        ram_addr,
        // RamReadEqRamWriteIfLoad
        ram_read - ram_write,
        // RamReadEqRdWriteIfLoad
        ram_read - rd_write,
        // Rs2EqRamWriteIfStore
        rs2 - ram_write,
        // LeftLookupZeroUnlessAddSubMul
        left_lookup,
        // LeftLookupEqLeftInputOtherwise
        left_lookup - left_input,
        // AssertLookupOne
        lookup_out - 1,
        // NextUnexpPCEqLookupIfShouldJump
        next_unexp_pc - lookup_out,
        // NextPCEqPCPlusOneIfInline
        next_pc - (pc + 1),
        // MustStartSequenceFromBeginning: DoNotUpdateUnexpandedPC - 1
        (row.flags[CircuitFlags::DoNotUpdateUnexpandedPC] as i128) - 1,
    ]
}

/// Evaluate Az for the second group
pub fn eval_az_second_group(row: &R1CSCycleInputs) -> [u8; NUM_REMAINING_R1CS_CONSTRAINTS] {
    use ConstraintName as N;
    let flags = &row.flags;
    let add = flags[CircuitFlags::AddOperands] as u8;
    let sub = flags[CircuitFlags::SubtractOperands] as u8;
    let mul = flags[CircuitFlags::MultiplyOperands] as u8;

    let mut out: [u8; NUM_REMAINING_R1CS_CONSTRAINTS] = [0u8; NUM_REMAINING_R1CS_CONSTRAINTS];
    let mut i = 0;
    while i < UNIFORM_R1CS_SECOND_GROUP.len() {
        let name = UNIFORM_R1CS_SECOND_GROUP[i].name;
        out[i] = match name {
            N::RamAddrEqRs1PlusImmIfLoadStore => {
                (flags[CircuitFlags::Load] || flags[CircuitFlags::Store]) as u8
            }
            N::RamAddrEqZeroIfNotLoadStore => {
                (!(flags[CircuitFlags::Load] || flags[CircuitFlags::Store])) as u8
            }
            N::RamReadEqRamWriteIfLoad => flags[CircuitFlags::Load] as u8,
            N::RamReadEqRdWriteIfLoad => flags[CircuitFlags::Load] as u8,
            N::Rs2EqRamWriteIfStore => flags[CircuitFlags::Store] as u8,
            N::LeftLookupZeroUnlessAddSubMul => add | sub | mul,
            N::LeftLookupEqLeftInputOtherwise => {
                !(flags[CircuitFlags::AddOperands]
                    || flags[CircuitFlags::SubtractOperands]
                    || flags[CircuitFlags::MultiplyOperands]) as u8
            }
            N::RightLookupAdd => flags[CircuitFlags::AddOperands] as u8,
            N::RightLookupSub => flags[CircuitFlags::SubtractOperands] as u8,
            N::RightLookupEqProductIfMul => flags[CircuitFlags::MultiplyOperands] as u8,
            N::RightLookupEqRightInputOtherwise => {
                !(flags[CircuitFlags::AddOperands]
                    || flags[CircuitFlags::SubtractOperands]
                    || flags[CircuitFlags::MultiplyOperands]
                    || flags[CircuitFlags::Advice]) as u8
            }
            N::AssertLookupOne => flags[CircuitFlags::Assert] as u8,
            N::RdWriteEqLookupIfWriteLookupToRd => row.write_lookup_output_to_rd_addr,
            N::RdWriteEqPCPlusConstIfWritePCtoRD => row.write_pc_to_rd_addr,
            N::NextUnexpPCEqLookupIfShouldJump => row.should_jump as u8,
            N::NextUnexpPCEqPCPlusImmIfShouldBranch => row.should_branch as u8,
            N::NextUnexpPCUpdateOtherwise => {
                let jump = flags[CircuitFlags::Jump] as u8;
                1u8.wrapping_sub(jump).wrapping_sub(row.should_branch as u8)
            }
            N::NextPCEqPCPlusOneIfInline => flags[CircuitFlags::VirtualInstruction] as u8,
            N::MustStartSequenceFromBeginning => 0u8,
        };
        i += 1;
    }
    out
}

/// Evaluate Bz for the second group
pub fn eval_bz_second_group(row: &R1CSCycleInputs) -> [S160; NUM_REMAINING_R1CS_CONSTRAINTS] {
    use ConstraintName as N;
    let mut out: [S160; NUM_REMAINING_R1CS_CONSTRAINTS] =
        [S160::zero(); NUM_REMAINING_R1CS_CONSTRAINTS];
    let mut i = 0;
    while i < UNIFORM_R1CS_SECOND_GROUP.len() {
        let name = UNIFORM_R1CS_SECOND_GROUP[i].name;
        out[i] = match name {
            N::RamAddrEqRs1PlusImmIfLoadStore => {
                let expected: i128 = if row.imm.is_positive {
                    (row.rs1_read_value as u128 + row.imm.magnitude_as_u64() as u128) as i128
                } else {
                    row.rs1_read_value as i128 - row.imm.magnitude_as_u64() as i128
                };
                S160::from(row.ram_addr as i128 - expected)
            }
            N::RamAddrEqZeroIfNotLoadStore => S160::from(row.ram_addr),
            N::RamReadEqRamWriteIfLoad => {
                S160::from_diff_u64(row.ram_read_value, row.ram_write_value)
            }
            N::RamReadEqRdWriteIfLoad => {
                S160::from_diff_u64(row.ram_read_value, row.rd_write_value)
            }
            N::Rs2EqRamWriteIfStore => S160::from_diff_u64(row.rs2_read_value, row.ram_write_value),
            N::LeftLookupZeroUnlessAddSubMul => S160::from(row.left_lookup),
            N::LeftLookupEqLeftInputOtherwise => {
                S160::from(row.left_lookup) - S160::from(row.left_input)
            }
            N::RightLookupAdd => {
                let expected_i128 = (row.left_input as i128) + row.right_input.to_i128();
                S160::from(row.right_lookup) - S160::from(expected_i128)
            }
            N::RightLookupSub => {
                let expected_i128 =
                    (row.left_input as i128) - row.right_input.to_i128() + (1i128 << 64);
                S160::from(row.right_lookup) - S160::from(expected_i128)
            }
            N::RightLookupEqProductIfMul => S160::from(row.right_lookup) - S160::from(row.product),
            N::RightLookupEqRightInputOtherwise => {
                S160::from(row.right_lookup) - S160::from(row.right_input)
            }
            N::AssertLookupOne => S160::from(row.lookup_output as i128 - 1),
            N::RdWriteEqLookupIfWriteLookupToRd => {
                S160::from_diff_u64(row.rd_write_value, row.lookup_output)
            }
            N::RdWriteEqPCPlusConstIfWritePCtoRD => {
                let const_term = 4 - if row.flags[CircuitFlags::IsCompressed] {
                    2
                } else {
                    0
                };
                S160::from(
                    row.rd_write_value as i128 - (row.unexpanded_pc as i128 + const_term as i128),
                )
            }
            N::NextUnexpPCEqLookupIfShouldJump => {
                S160::from_diff_u64(row.next_unexpanded_pc, row.lookup_output)
            }
            N::NextUnexpPCEqPCPlusImmIfShouldBranch => S160::from(
                row.next_unexpanded_pc as i128 - (row.unexpanded_pc as i128 + row.imm.to_i128()),
            ),
            N::NextUnexpPCUpdateOtherwise => {
                let const_term =
                    4 - if row.flags[CircuitFlags::DoNotUpdateUnexpandedPC] {
                        4
                    } else {
                        0
                    } - if row.flags[CircuitFlags::IsCompressed] {
                        2
                    } else {
                        0
                    };
                let target = row.unexpanded_pc as i128 + const_term;
                S160::from(row.next_unexpanded_pc as i128 - target)
            }
            N::NextPCEqPCPlusOneIfInline => S160::from(row.next_pc as i128 - (row.pc as i128 + 1)),
            N::MustStartSequenceFromBeginning => S160::zero(),
        };
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use strum::IntoEnumIterator;

    /// Test that the constraint name enum matches the uniform R1CS order.
    #[test]
    fn constraint_enum_matches_uniform_r1cs_order() {
        let enum_order: Vec<ConstraintName> = ConstraintName::iter().collect();
        let array_order: Vec<ConstraintName> = UNIFORM_R1CS.iter().map(|nc| nc.name).collect();
        assert_eq!(array_order.len(), NUM_R1CS_CONSTRAINTS);
        assert_eq!(enum_order, array_order);
    }

    /// Test that all Cz terms in UNIFORM_R1CS are zero.
    /// We currently only use conditional equality constraints
    /// of the form Az * Bz = 0, which means Cz must be identically zero.
    #[test]
    fn all_cz_terms_are_zero() {
        for (i, named_constraint) in UNIFORM_R1CS.iter().enumerate() {
            let c = &named_constraint.cons.c;

            // Check that the C LC is structurally Zero
            assert!(
                matches!(c, LC::Zero),
                "Constraint {} ({:?}) has non-zero Cz: the C term must be LC::Zero but got {:?}",
                i,
                named_constraint.name,
                c
            );

            // Double-check: verify it has no terms and no constant
            assert_eq!(
                c.num_terms(),
                0,
                "Constraint {} ({:?}) has {} terms in Cz, expected 0",
                i,
                named_constraint.name,
                c.num_terms()
            );

            assert!(
                c.const_term().is_none(),
                "Constraint {} ({:?}) has a constant term in Cz, expected None",
                i,
                named_constraint.name
            );
        }
    }
}
