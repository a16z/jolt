//! Compile-time constant R1CS constraints and grouping metadata
//!
//! This module defines the static, compile-time representation of all uniform
//! R1CS constraints used by the zkVM, along with constants and tables that split
//! the constraints into two groups optimized for the univariate-skip protocol.
//!
//! Evaluation logic (how `Az`/`Bz` are computed and folded) now lives in
//! `r1cs::evaluation`. Use the typed evaluators `R1CSFirstGroup` and
//! `R1CSSecondGroup` in that module to compute guards/magnitudes and to fold
//! them against window weights.
//!
//! Grouping overview:
//! - Group 0 (first group) contains `UNIVARIATE_SKIP_DOMAIN_SIZE = ceil(N/2)`
//!   constraints with boolean `Az` and small-width `Bz`.
//! - Group 1 (second group) is the complement with nonnegative `Az` (e.g. `u8`)
//!   and wider `Bz` arithmetic.
//!
//! ## Adding a new constraint
//!
//! 1. Add a new variant to `ConstraintName` (keep the same order as `UNIFORM_R1CS`).
//! 2. Add the constraint to `UNIFORM_R1CS` using the `r1cs_eq_conditional!` macro.
//! 3. Assign the constraint to a group:
//!    - Put its name in `UNIFORM_R1CS_FIRST_GROUP_NAMES` if it matches Group 0
//!      characteristics (boolean guards, ~64-bit `Bz`).
//!    - Otherwise it will appear in Group 1 automatically as the complement.
//! 4. Maintain the grouping invariant: `UNIFORM_R1CS_FIRST_GROUP_NAMES.len()` must equal
//!    `UNIVARIATE_SKIP_DOMAIN_SIZE = ceil(NUM_R1CS_CONSTRAINTS/2)`; the first group is
//!    never smaller than the second.
//! 5. If the new constraint changes the shapes of guards/magnitudes, update the
//!    evaluators in `r1cs::evaluation` accordingly (`Az*/Bz*` structs and methods).
//!
//! ## Removing a constraint
//!
//! 1. Remove it from `UNIFORM_R1CS`.
//! 2. Remove the corresponding variant from `ConstraintName`.
//! 3. If present, remove its name from `UNIFORM_R1CS_FIRST_GROUP_NAMES`.
//! 4. Re-check that `UNIFORM_R1CS_FIRST_GROUP_NAMES.len()` equals
//!    `UNIVARIATE_SKIP_DOMAIN_SIZE` after the change; adjust the first group
//!    selection to preserve the invariant that the first group is never smaller
//!    than the second.
//! 5. If evaluation shapes are affected, update `r1cs::evaluation`.
//!
//! ## Grouping guidance
//!
//! - Prefer Group 0 for boolean `Az` and `Bz` that can be represented in ~64 bits.
//! - Prefer Group 1 when `Az` are small nonnegative integers and `Bz` require
//!   wider arithmetic.
//! - This split minimizes conversions and maximizes accumulator efficiency.

use super::inputs::JoltR1CSInputs;
use crate::zkvm::instruction::CircuitFlags;
use strum::EnumCount;
use strum_macros::{EnumCount, EnumIter};

pub use super::ops::{Term, LC};

/// A single R1CS constraint row
#[derive(Clone, Copy, Debug)]
pub struct Constraint {
    pub a: LC,
    pub b: LC,
    // No c needed for now, all eq-conditional constraints
    // pub c: LC,
}

impl Constraint {
    pub const fn new(a: LC, b: LC) -> Self {
        Self { a, b }
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
    //     assert!(1 == DoNotUpdateUnexpandedPC)
    // }
    // (note: we write the constraint in this form to keep Bz boolean)
    r1cs_eq_conditional!(
        name: ConstraintName::MustStartSequenceFromBeginning,
        if { { JoltR1CSInputs::NextIsVirtual } - { JoltR1CSInputs::NextIsFirstInSequence } }
        => ( { 1i128 } ) == ( { JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC) } )
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

/// Degree of the first-round polynomial.
pub const FIRST_ROUND_POLY_DEGREE_BOUND: usize = FIRST_ROUND_POLY_NUM_COEFFS - 1;

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
        cons: Constraint::new(LC::zero(), LC::zero()),
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
