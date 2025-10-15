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
//! 3. Optionally (but encouraged) add custom evaluators in `eval_az_by_name` and `eval_bz_by_name`
//! 4. Update `NUM_R1CS_CONSTRAINTS`
//!
//! ## Removing a constraint
//!
//! To remove an R1CS constraint:
//! 1. Remove the constraint from `UNIFORM_R1CS` array
//! 2. Remove the corresponding variant from `ConstraintName` enum
//! 3. Remove any custom evaluator from `eval_az_bz_by_name`
//! 4. Update `NUM_R1CS_CONSTRAINTS`
//!
//! ## Custom evaluators
//!
//! Custom evaluators in `eval_az_by_name`/`eval_bz_by_name` provide optimized Az/Bz evaluation
//! using `SmallScalar` types to avoid field conversions. They should:
//! - Use appropriate `I8OrI96` variants (prefer `Bool` or `I8` for flags/small sums)
//! - Use appropriate `S160` variants (prefer `U64AndSign` for u64-u64 diffs)
//! - Only use `U128AndSign` when 128-bit arithmetic is inherently required

use super::inputs::{JoltR1CSInputs, R1CSCycleInputs};
use crate::field::JoltField;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::zkvm::instruction::CircuitFlags;
use ark_ff::biginteger::{I8OrI96, S160};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, strum_macros::EnumIter)]
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
}

#[derive(Clone, Copy, Debug)]
pub struct NamedConstraint {
    pub name: ConstraintName,
    pub cons: Constraint,
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

// Removed r1cs_prod! macro; not used anymore.

#[macro_export]
macro_rules! r1cs_eq_named {
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

// Removed r1cs_if_else_named! macro; not used anymore.

// Removed r1cs_prod_named! macro; not used anymore.

// Constants to be used in Spartan

/// Number of uniform R1CS constraints
pub const NUM_R1CS_CONSTRAINTS: usize = 18;
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

/// Static table of all R1CS uniform constraints.
///
/// All constraints are eq-conditional of the form: if {cond} => (left == right).
/// There are no if-else or product constraints; consequently the Cz term is
/// identically zero across the board.
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
    // If MultiplyOperands {
    //     assert!(RightLookupOperand == Product)
    // }
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
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::InlineSequenceInstruction) } }
        => ( { JoltR1CSInputs::NextPC } ) == ( { JoltR1CSInputs::PC } + { 1i128 } )
    ),
];

// =============================================================================
// Filtered views of UNIFORM_R1CS
// =============================================================================
/// Order-preserving, compile-time filter over `UNIFORM_R1CS` by constraint names.
///
/// This allows us to build curated, documented subsets without duplicating the
/// constraint definitions or changing their relative order.
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

/// Select constraints from `UNIFORM_R1CS` whose names appear in `names`, preserving
/// original order. Panics at compile time if any requested name is missing.
pub const fn filter_uniform_r1cs<const N: usize>(
    names: &[ConstraintName; N],
) -> [NamedConstraint; N] {
    // Initialize with a dummy; will be overwritten for all positions 0..N-1
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

/// Compute the complement of `UNIFORM_R1CS_FIRST_GROUP_NAMES` within `UNIFORM_R1CS`,
/// preserving the global order. Returns exactly 13 names.
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

/// UNIFORM_R1CS_FIRST_GROUP_NAMES: 14 boolean-guarded equality constraints with Cz=Zero
/// and Bz bounded in i128 under `R1CSCycleInputs` semantics (u64 values and S64 immediates).
///
/// Rationale (documented here rather than in the identifier):
/// - Cz is always zero (all are eq-conditional constraints)
/// - Az is a boolean selector (flag-derived or derived boolean)
/// - Bz is a S160
///
/// Selection policy: the matching constraints in `UNIFORM_R1CS`, order-preserving.
pub const UNIFORM_R1CS_FIRST_GROUP_NAMES: [ConstraintName; UNIVARIATE_SKIP_DOMAIN_SIZE] = [
    ConstraintName::RamReadEqRamWriteIfLoad,
    ConstraintName::RamReadEqRdWriteIfLoad,
    ConstraintName::Rs2EqRamWriteIfStore,
    ConstraintName::RightLookupAdd,
    ConstraintName::RightLookupSub,
    ConstraintName::AssertLookupOne,
    ConstraintName::NextUnexpPCEqLookupIfShouldJump,
    ConstraintName::RightLookupEqRightInputOtherwise,
    ConstraintName::LeftLookupEqLeftInputOtherwise,
];

/// UNIFORM_R1CS_SECOND_GROUP_NAMES: computed complement of `UNIFORM_R1CS_FIRST_GROUP_NAMES`
/// within `UNIFORM_R1CS`, order-preserving.
pub const UNIFORM_R1CS_SECOND_GROUP_NAMES: [ConstraintName; NUM_REMAINING_R1CS_CONSTRAINTS] =
    complement_first_group_names();

/// 14 boolean-guarded eq constraints with Cz=Zero and i128-bounded Bz semantics.
pub static UNIFORM_R1CS_FIRST_GROUP: [NamedConstraint; UNIVARIATE_SKIP_DOMAIN_SIZE] =
    filter_uniform_r1cs(&UNIFORM_R1CS_FIRST_GROUP_NAMES);

/// Remaining 13 constraints (complement of `UNIFORM_R1CS_FIRST_GROUP` within `UNIFORM_R1CS`).
pub static UNIFORM_R1CS_SECOND_GROUP: [NamedConstraint; NUM_REMAINING_R1CS_CONSTRAINTS] =
    filter_uniform_r1cs(&UNIFORM_R1CS_SECOND_GROUP_NAMES);

/// Evaluate Az by name using a fully materialized R1CS cycle inputs
pub fn eval_az_by_name(c: &NamedConstraint, row: &R1CSCycleInputs) -> I8OrI96 {
    use ConstraintName as N;
    match c.name {
        N::RamAddrEqRs1PlusImmIfLoadStore => {
            // Az: Load OR Store flag (0/1)
            (row.flags[CircuitFlags::Load] || row.flags[CircuitFlags::Store]).into()
        }
        N::RamAddrEqZeroIfNotLoadStore => {
            let cond = !(row.flags[CircuitFlags::Load] || row.flags[CircuitFlags::Store]);
            I8OrI96::from(cond)
        }
        // Az: Load flag (0/1)
        N::RamReadEqRamWriteIfLoad => row.flags[CircuitFlags::Load].into(),
        // Az: Load flag (0/1)
        N::RamReadEqRdWriteIfLoad => row.flags[CircuitFlags::Load].into(),
        // Az: Store flag (0/1)
        N::Rs2EqRamWriteIfStore => row.flags[CircuitFlags::Store].into(),
        N::LeftLookupZeroUnlessAddSubMul => {
            // NOTE: these are exclusive circuit flags (validated in tests)
            let add = row.flags[CircuitFlags::AddOperands];
            let sub = row.flags[CircuitFlags::SubtractOperands];
            let mul = row.flags[CircuitFlags::MultiplyOperands];
            (add || sub || mul).into()
        }
        N::LeftLookupEqLeftInputOtherwise => {
            let add = row.flags[CircuitFlags::AddOperands];
            let sub = row.flags[CircuitFlags::SubtractOperands];
            let mul = row.flags[CircuitFlags::MultiplyOperands];
            (!(add || sub || mul)).into()
        }
        // Az: AddOperands flag (0/1)
        N::RightLookupAdd => row.flags[CircuitFlags::AddOperands].into(),
        // Az: SubtractOperands flag (0/1)
        N::RightLookupSub => row.flags[CircuitFlags::SubtractOperands].into(),
        // Az: MultiplyOperands flag (0/1)
        N::RightLookupEqProductIfMul => row.flags[CircuitFlags::MultiplyOperands].into(),
        N::RightLookupEqRightInputOtherwise => {
            // NOTE: relies on exclusivity of circuit flags (validated in tests):
            // return 1 only if none of add/sub/mul/adv is set
            let add = row.flags[CircuitFlags::AddOperands];
            let sub = row.flags[CircuitFlags::SubtractOperands];
            let mul = row.flags[CircuitFlags::MultiplyOperands];
            let adv = row.flags[CircuitFlags::Advice];
            (!(add || sub || mul || adv)).into()
        }
        // Az: Assert flag (0/1)
        N::AssertLookupOne => row.flags[CircuitFlags::Assert].into(),
        // Az: WriteLookupOutputToRD indicator (0/1)
        N::RdWriteEqLookupIfWriteLookupToRd => {
            I8OrI96::from_i8(row.write_lookup_output_to_rd_addr as i8)
        }
        // Az: WritePCtoRD indicator (0/1)
        N::RdWriteEqPCPlusConstIfWritePCtoRD => I8OrI96::from_i8(row.write_pc_to_rd_addr as i8),
        // Az: ShouldJump indicator (0/1)
        N::NextUnexpPCEqLookupIfShouldJump => row.should_jump.into(),
        // Note: Az uses ShouldBranch in the u64 domain (product Branch * LookupOutput)
        // Az: ShouldBranch indicator (0/1)
        N::NextUnexpPCEqPCPlusImmIfShouldBranch => I8OrI96::from(row.should_branch),
        N::NextUnexpPCUpdateOtherwise => {
            // Az encodes 1 - ShouldBranch - Jump = (1 - Jump) - ShouldBranch.
            let jump = row.flags[CircuitFlags::Jump];
            let not_jump: i128 = if jump { 0 } else { 1 };
            let diff = not_jump - (row.should_branch as i128);
            I8OrI96::from(diff)
        }
        // Az: InlineSequenceInstruction flag (0/1)
        N::NextPCEqPCPlusOneIfInline => row.flags[CircuitFlags::InlineSequenceInstruction].into(),
    }
}

/// Evaluate Bz by name using a fully materialized R1CS cycle inputs
pub fn eval_bz_by_name(c: &NamedConstraint, row: &R1CSCycleInputs) -> S160 {
    use ConstraintName as N;
    match c.name {
        N::RamAddrEqRs1PlusImmIfLoadStore => {
            // B: RamAddress - (Rs1Value + Imm) (i128 arithmetic)
            let expected: i128 = if row.imm.is_positive {
                (row.rs1_read_value as u128 + row.imm.magnitude_as_u64() as u128) as i128
            } else {
                row.rs1_read_value as i128 - row.imm.magnitude_as_u64() as i128
            };
            S160::from(row.ram_addr as i128 - expected)
        }
        N::RamAddrEqZeroIfNotLoadStore => {
            // B: RamAddress - 0
            S160::from(row.ram_addr)
        }
        // B: RamReadValue - RamWriteValue (u64 bit-pattern difference)
        N::RamReadEqRamWriteIfLoad => S160::from_diff_u64(row.ram_read_value, row.ram_write_value),
        // B: RamReadValue - RdWriteValue (u64 bit-pattern difference)
        N::RamReadEqRdWriteIfLoad => S160::from_diff_u64(row.ram_read_value, row.rd_write_value),
        // B: Rs2Value - RamWriteValue (u64 bit-pattern difference)
        N::Rs2EqRamWriteIfStore => S160::from_diff_u64(row.rs2_read_value, row.ram_write_value),
        // B: LeftLookupOperand - 0
        N::LeftLookupZeroUnlessAddSubMul => S160::from(row.left_lookup),
        N::LeftLookupEqLeftInputOtherwise => {
            // B: LeftLookupOperand - LeftInstructionInput
            S160::from(row.left_lookup) - S160::from(row.left_input)
        }
        N::RightLookupAdd => {
            // B: RightLookupOperand - (LeftInstructionInput + RightInstructionInput) with full-width integer semantics
            let expected_i128 = (row.left_input as i128) + row.right_input.to_i128();
            S160::from(row.right_lookup) - S160::from(expected_i128)
        }
        N::RightLookupSub => {
            // B: RightLookupOperand - (LeftInstructionInput - RightInstructionInput + 2^64)
            // with full-width integer semantics (matches the +2^64 in the uniform constraint)
            let expected_i128 =
                (row.left_input as i128) - row.right_input.to_i128() + (1i128 << 64);
            S160::from(row.right_lookup) - S160::from(expected_i128)
        }
        N::RightLookupEqProductIfMul => {
            // B: RightLookupOperand - Product with full 128-bit semantics
            S160::from(row.right_lookup) - S160::from(row.product)
        }
        N::RightLookupEqRightInputOtherwise => {
            // B: RightLookupOperand - RightInstructionInput with exact integer semantics
            S160::from(row.right_lookup) - S160::from(row.right_input)
        }
        // B: LookupOutput - 1 (i128 arithmetic)
        N::AssertLookupOne => S160::from(row.lookup_output as i128 - 1),
        N::RdWriteEqLookupIfWriteLookupToRd => {
            // B: RdWriteValue - LookupOutput (u64 bit-pattern difference)
            S160::from_diff_u64(row.rd_write_value, row.lookup_output)
        }
        N::RdWriteEqPCPlusConstIfWritePCtoRD => {
            // B: RdWriteValue - (UnexpandedPC + (4 - 2*IsCompressed)) (i128 arithmetic)
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
            // Note: B uses u64 bit-pattern difference here
            // B: NextUnexpandedPC - LookupOutput (i128 arithmetic)
            S160::from_diff_u64(row.next_unexpanded_pc, row.lookup_output)
        }
        // B: NextUnexpandedPC - (UnexpandedPC + Imm) (i128 arithmetic)
        N::NextUnexpPCEqPCPlusImmIfShouldBranch => S160::from(
            row.next_unexpanded_pc as i128 - (row.unexpanded_pc as i128 + row.imm.to_i128()),
        ),
        N::NextUnexpPCUpdateOtherwise => {
            // B: NextUnexpandedPC - target, where target = UnexpandedPC + 4 - 4*DoNotUpdateUnexpandedPC - 2*IsCompressed (i128 arithmetic)
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
        N::NextPCEqPCPlusOneIfInline => {
            // B: NextPC - (PC + 1) (i128 arithmetic)
            S160::from(row.next_pc as i128 - (row.pc as i128 + 1))
        }
    }
}

/// Batched evaluation using a fully materialized R1CS cycle inputs. This avoids any repeated
/// reads from the trace or bytecode and computes all constraints.
pub fn eval_az_bz_batch_from_row(
    constraints: &[NamedConstraint],
    row: &R1CSCycleInputs,
    az_output: &mut [I8OrI96],
    bz_output: &mut [S160],
) {
    assert_eq!(constraints.len(), az_output.len());
    assert_eq!(constraints.len(), bz_output.len());
    for (i, constraint) in constraints.iter().enumerate() {
        az_output[i] = eval_az_by_name(constraint, row);
        bz_output[i] = eval_bz_by_name(constraint, row);
    }
}

/// Evaluate Cz by name using a fully materialized R1CS cycle inputs
/// Evaluate Az for the first group as booleans in the group order (length UNIVARIATE_SKIP_DOMAIN_SIZE).
/// Booleans are derived from the same semantics as `eval_az_by_name`, with non-zero selectors
/// interpreted as true.
pub fn eval_az_first_group(row: &R1CSCycleInputs) -> [bool; UNIVARIATE_SKIP_DOMAIN_SIZE] {
    // Order matches UNIFORM_R1CS_FIRST_GROUP_NAMES (now 9 entries)
    [
        row.flags[CircuitFlags::Load],             // RamReadEqRamWriteIfLoad
        row.flags[CircuitFlags::Load],             // RamReadEqRdWriteIfLoad
        row.flags[CircuitFlags::Store],            // Rs2EqRamWriteIfStore
        row.flags[CircuitFlags::AddOperands],      // RightLookupAdd
        row.flags[CircuitFlags::SubtractOperands], // RightLookupSub
        row.flags[CircuitFlags::Assert],           // AssertLookupOne
        row.should_jump,                           // NextUnexpPCEqLookupIfShouldJump
        {
            let add = row.flags[CircuitFlags::AddOperands];
            let sub = row.flags[CircuitFlags::SubtractOperands];
            let mul = row.flags[CircuitFlags::MultiplyOperands];
            let adv = row.flags[CircuitFlags::Advice];
            !(add || sub || mul || adv)
        }, // RightLookupEqRightInputOtherwise
        {
            let add = row.flags[CircuitFlags::AddOperands];
            let sub = row.flags[CircuitFlags::SubtractOperands];
            let mul = row.flags[CircuitFlags::MultiplyOperands];
            !(add || sub || mul)
        }, // LeftLookupEqLeftInputOtherwise
    ]
}

/// Evaluate Bz for the first group as S160 in the group order (length UNIVARIATE_SKIP_DOMAIN_SIZE),
/// using the same semantics as `eval_bz_by_name`.
pub fn eval_bz_first_group(row: &R1CSCycleInputs) -> [S160; UNIVARIATE_SKIP_DOMAIN_SIZE] {
    let mut out: [S160; UNIVARIATE_SKIP_DOMAIN_SIZE] = [S160::zero(); UNIVARIATE_SKIP_DOMAIN_SIZE];
    let mut i = 0;
    while i < UNIFORM_R1CS_FIRST_GROUP.len() {
        out[i] = eval_bz_by_name(&UNIFORM_R1CS_FIRST_GROUP[i], row);
        i += 1;
    }
    out
}

/// Evaluate Az for the second group in group order, using the same small-scalar
/// semantics as `eval_az_by_name`.
pub fn eval_az_second_group(row: &R1CSCycleInputs) -> [I8OrI96; NUM_REMAINING_R1CS_CONSTRAINTS] {
    let mut out: [I8OrI96; NUM_REMAINING_R1CS_CONSTRAINTS] =
        [I8OrI96::zero(); NUM_REMAINING_R1CS_CONSTRAINTS];
    let mut i = 0;
    while i < UNIFORM_R1CS_SECOND_GROUP.len() {
        out[i] = eval_az_by_name(&UNIFORM_R1CS_SECOND_GROUP[i], row);
        i += 1;
    }
    out
}

/// Evaluate Bz for the second group in group order, using the same small-scalar
/// semantics as `eval_bz_by_name`.
pub fn eval_bz_second_group(row: &R1CSCycleInputs) -> [S160; NUM_REMAINING_R1CS_CONSTRAINTS] {
    let mut out: [S160; NUM_REMAINING_R1CS_CONSTRAINTS] =
        [S160::zero(); NUM_REMAINING_R1CS_CONSTRAINTS];
    let mut i = 0;
    while i < UNIFORM_R1CS_SECOND_GROUP.len() {
        out[i] = eval_bz_by_name(&UNIFORM_R1CS_SECOND_GROUP[i], row);
        i += 1;
    }
    out
}

/// Evaluate Cz for the second group in group order.
/// Returns S160 values only for constraints whose Cz is non-zero; returns zero for the rest.
// (intentionally no Cz second-group evaluator; Cz is not needed for the groups interface)

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
}
