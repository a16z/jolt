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

use super::inputs::{JoltR1CSInputs, WitnessRowAccessor};
use crate::field::JoltField;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::zkvm::instruction::CircuitFlags;
// use crate::utils::small_scalar::SmallScalar;
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

// =============================================================================
// Streaming accessor evaluation for LC
// =============================================================================
impl LC {
    #[inline]
    pub fn evaluate_row_with<F: JoltField>(
        &self,
        accessor: &dyn WitnessRowAccessor<F, JoltR1CSInputs>,
        row: usize,
    ) -> F {
        let mut result = F::zero();
        self.for_each_term(|input_index, coeff| {
            result += crate::utils::small_scalar::SmallScalar::field_mul(
                &coeff,
                accessor.value_at_field(JoltR1CSInputs::from_index(input_index), row),
            );
        });
        if let Some(c) = self.const_term() {
            result += crate::utils::small_scalar::SmallScalar::to_field::<F>(c);
        }
        result
    }

    #[inline]
    pub fn evaluate_row_with_old<F: JoltField>(
        &self,
        accessor: &dyn WitnessRowAccessor<F, JoltR1CSInputs>,
        row: usize,
    ) -> F {
        let mut result = F::zero();
        self.for_each_term(|input_index, coeff| {
            let v = accessor.value_at_field(JoltR1CSInputs::from_index(input_index), row);
            result += crate::utils::small_scalar::SmallScalar::field_mul(&coeff, v);
        });
        if let Some(c) = self.const_term() {
            result += crate::utils::small_scalar::SmallScalar::to_field::<F>(c);
        }
        result
    }
}

// =============================================================================
// CONSTRAINT BUILDER FUNCTIONS
// =============================================================================
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

// =============================================================================
// Named constraints with minimal Cz marker
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, strum_macros::EnumIter)]
pub enum ConstraintName {
    LeftInputEqRs1,
    LeftInputEqPC,
    LeftInputZeroOtherwise,
    RightInputEqRs2,
    RightInputEqImm,
    RightInputZeroOtherwise,
    RamAddrEqRs1PlusImmIfLoadStore,
    RamReadEqRamWriteIfLoad,
    RamReadEqRdWriteIfLoad,
    Rs2EqRamWriteIfStore,
    LeftLookupZeroUnlessAddSubMul,
    RightLookupAdd,
    RightLookupSub,
    ProductDef,
    RightLookupEqProductIfMul,
    RightLookupEqRightInputOtherwise,
    AssertLookupOne,
    WriteLookupOutputToRDDef,
    RdWriteEqLookupIfWriteLookupToRd,
    WritePCtoRDDef,
    RdWriteEqPCPlusConstIfWritePCtoRD,
    ShouldJumpDef,
    NextUnexpPCEqLookupIfShouldJump,
    ShouldBranchDef,
    NextUnexpPCEqPCPlusImmIfShouldBranch,
    NextUnexpPCUpdateOtherwise,
    NextPCEqPCPlusOneIfInline,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CzKind {
    Zero,
    NonZero,
}

#[derive(Clone, Copy, Debug)]
pub struct NamedConstraint {
    pub name: ConstraintName,
    pub cons: Constraint,
    pub cz: CzKind,
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
            cz: $crate::zkvm::r1cs::constraints::CzKind::Zero,
        }
    }};
}

/// r1cs_if_else!: verbose if-then-else with explicit result
///
/// Usage: `r1cs_if_else!(name: ConstraintName::Foo, if { COND } => { TRUE } else { FALSE } => { RESULT });`
#[macro_export]
macro_rules! r1cs_if_else {
    (name: $nm:expr, if { $($cond:tt)* } => ( $($tval:tt)* ) else ( $($fval:tt)* ) => ( $($result:tt)* ) ) => {{
        $crate::zkvm::r1cs::constraints::NamedConstraint {
            name: $nm,
            cons: $crate::zkvm::r1cs::constraints::constraint_if_else_lc(
                $crate::lc!($($cond)*),
                $crate::lc!($($tval)*),
                $crate::lc!($($fval)*),
                $crate::lc!($($result)*),
            ),
            cz: $crate::zkvm::r1cs::constraints::CzKind::NonZero,
        }
    }};
}

/// r1cs_prod!: product constraint
///
/// Usage: `r1cs_prod!(name: ConstraintName::Foo, { LEFT } * { RIGHT } == { RESULT });`
#[macro_export]
macro_rules! r1cs_prod {
    (name: $nm:expr, ( $($left:tt)* ) * ( $($right:tt)* ) == ( $($result:tt)* ) ) => {{
        $crate::zkvm::r1cs::constraints::NamedConstraint {
            name: $nm,
            cons: $crate::zkvm::r1cs::constraints::constraint_prod_lc(
                $crate::lc!($($left)*),
                $crate::lc!($($right)*),
                $crate::lc!($($result)*),
            ),
            cz: $crate::zkvm::r1cs::constraints::CzKind::NonZero,
        }
    }};
}

// ==========================
// Named macro variants
// ==========================

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
            cz: $crate::zkvm::r1cs::constraints::CzKind::Zero,
        }
    }};
}

#[macro_export]
macro_rules! r1cs_if_else_named {
    (name: $nm:expr, if { $($cond:tt)* } => ( $($tval:tt)* ) else ( $($fval:tt)* ) => ( $($result:tt)* ) ) => {{
        $crate::zkvm::r1cs::constraints::NamedConstraint {
            name: $nm,
            cons: $crate::zkvm::r1cs::constraints::constraint_if_else_lc(
                $crate::lc!($($cond)*),
                $crate::lc!($($tval)*),
                $crate::lc!($($fval)*),
                $crate::lc!($($result)*),
            ),
            cz: $crate::zkvm::r1cs::constraints::CzKind::NonZero,
        }
    }};
}

#[macro_export]
macro_rules! r1cs_prod_named {
    (name: $nm:expr, ( $($left:tt)* ) * ( $($right:tt)* ) == ( $($result:tt)* ) ) => {{
        $crate::zkvm::r1cs::constraints::NamedConstraint {
            name: $nm,
            cons: $crate::zkvm::r1cs::constraints::constraint_prod_lc(
                $crate::lc!($($left)*),
                $crate::lc!($($right)*),
                $crate::lc!($($result)*),
            ),
            cz: $crate::zkvm::r1cs::constraints::CzKind::NonZero,
        }
    }};
}

/// Number of uniform R1CS constraints
pub const NUM_R1CS_CONSTRAINTS: usize = 27;

/// Static table of all 28 R1CS uniform constraints.
pub static UNIFORM_R1CS: [NamedConstraint; NUM_R1CS_CONSTRAINTS] = [
    // if LeftOperandIsRs1Value { assert!(LeftInstructionInput == Rs1Value) }
    r1cs_eq_conditional!(
        name: ConstraintName::LeftInputEqRs1,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsRs1Value) } }
        => ( { JoltR1CSInputs::LeftInstructionInput } ) == ( { JoltR1CSInputs::Rs1Value } )
    ),
    // if LeftOperandIsPC { assert!(LeftInstructionInput == UnexpandedPC) }
    r1cs_eq_conditional!(
        name: ConstraintName::LeftInputEqPC,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsPC) } }
        => ( { JoltR1CSInputs::LeftInstructionInput } ) == ( { JoltR1CSInputs::UnexpandedPC } )
    ),
    // if !(LeftOperandIsRs1Value || LeftOperandIsPC)  {
    //     assert!(LeftInstructionInput == 0)
    // }
    // Note that LeftOperandIsRs1Value and LeftOperandIsPC are mutually exclusive flags
    r1cs_eq_conditional!(
        name: ConstraintName::LeftInputZeroOtherwise,
        if { { 1i128 } - { JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsRs1Value) } - { JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsPC) } }
        => ( { JoltR1CSInputs::LeftInstructionInput } ) == ( { 0i128 } )
    ),
    // if RightOperandIsRs2Value { assert!(RightInstructionInput == Rs2Value) }
    r1cs_eq_conditional!(
        name: ConstraintName::RightInputEqRs2,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsRs2Value) } }
        => ( { JoltR1CSInputs::RightInstructionInput } ) == ( { JoltR1CSInputs::Rs2Value } )
    ),
    // if RightOperandIsImm { assert!(RightInstructionInput == Imm) }
    r1cs_eq_conditional!(
        name: ConstraintName::RightInputEqImm,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsImm) } }
        => ( { JoltR1CSInputs::RightInstructionInput } ) == ( { JoltR1CSInputs::Imm } )
    ),
    // if !(RightOperandIsRs2Value || RightOperandIsImm)  {
    //     assert!(RightInstructionInput == 0)
    // }
    // Note that RightOperandIsRs2Value and RightOperandIsImm are mutually exclusive flags
    r1cs_eq_conditional!(
        name: ConstraintName::RightInputZeroOtherwise,
        if { { 1i128 } - { JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsRs2Value) } - { JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsImm) } }
        => ( { JoltR1CSInputs::RightInstructionInput } ) == ( { 0i128 } )
    ),
    // if Load || Store {
    //     assert!(RamAddress == Rs1Value + Imm)
    // } else {
    //     assert!(RamAddress == 0)
    // }
    r1cs_if_else!(
        name: ConstraintName::RamAddrEqRs1PlusImmIfLoadStore,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::Load) } + { JoltR1CSInputs::OpFlags(CircuitFlags::Store) } }
        => ( { JoltR1CSInputs::Rs1Value } + { JoltR1CSInputs::Imm } )
        else ( { 0i128 } )
        => ( { JoltR1CSInputs::RamAddress } )
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
    r1cs_if_else!(
        name: ConstraintName::LeftLookupZeroUnlessAddSubMul,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands) } + { JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands) } + { JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands) } }
        => ( { 0i128 } )
        else ( { JoltR1CSInputs::LeftInstructionInput } )
        => ( { JoltR1CSInputs::LeftLookupOperand } )
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
    // if MultiplyOperands {
    //     assert!(RightLookupOperand == Rs1Value * Rs2Value)
    // }
    r1cs_prod!(
        name: ConstraintName::ProductDef,
        ({ JoltR1CSInputs::LeftInstructionInput }) * ({ JoltR1CSInputs::RightInstructionInput })
            == ({ JoltR1CSInputs::Product })
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
    r1cs_prod!(
        name: ConstraintName::WriteLookupOutputToRDDef,
        ({ JoltR1CSInputs::Rd })
            * ({ JoltR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToRD) })
            == ({ JoltR1CSInputs::WriteLookupOutputToRD })
    ),
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
    r1cs_prod!(
        name: ConstraintName::WritePCtoRDDef,
        ({ JoltR1CSInputs::Rd }) * ({ JoltR1CSInputs::OpFlags(CircuitFlags::Jump) })
            == ({ JoltR1CSInputs::WritePCtoRD })
    ),
    r1cs_eq_conditional!(
        name: ConstraintName::RdWriteEqPCPlusConstIfWritePCtoRD,
        if { { JoltR1CSInputs::WritePCtoRD } }
        => ( { JoltR1CSInputs::RdWriteValue } ) == ( { JoltR1CSInputs::UnexpandedPC } + { 4i128 } - { 2 * JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed) } )
    ),
    // if Jump && !NextIsNoop {
    //     assert!(NextUnexpandedPC == LookupOutput)
    // }
    r1cs_prod!(
        name: ConstraintName::ShouldJumpDef,
        ({ JoltR1CSInputs::OpFlags(CircuitFlags::Jump) })
            * ({ 1i128 } - { JoltR1CSInputs::NextIsNoop })
            == ({ JoltR1CSInputs::ShouldJump })
    ),
    r1cs_eq_conditional!(
        name: ConstraintName::NextUnexpPCEqLookupIfShouldJump,
        if { { JoltR1CSInputs::ShouldJump } }
        => ( { JoltR1CSInputs::NextUnexpandedPC } ) == ( { JoltR1CSInputs::LookupOutput } )
    ),
    // if Branch && LookupOutput {
    //     assert!(NextUnexpandedPC == UnexpandedPC + Imm)
    // }
    r1cs_prod!(
        name: ConstraintName::ShouldBranchDef,
        ({ JoltR1CSInputs::OpFlags(CircuitFlags::Branch) }) * ({ JoltR1CSInputs::LookupOutput })
            == ({ JoltR1CSInputs::ShouldBranch })
    ),
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

#[inline]
fn diff_to_bz(diff: i128) -> S160 {
    // New S160 is a sign-magnitude (64*2 + 32)-bit type; From<i128> fits directly
    S160::from(diff)
}

// =============================
// Precise integer helpers (no clamping)
// =============================

#[inline]
fn s160_from_u64_diff(a: u64, b: u64) -> S160 {
    let mag = a.abs_diff(b);
    let is_positive = a >= b;
    S160::new([mag, 0], 0, is_positive)
}

#[inline]
fn s160_from_u128_mag(mag: u128, is_positive: bool) -> S160 {
    S160::new([mag as u64, (mag >> 64) as u64], 0, is_positive)
}

#[inline]
fn u128_diff_to_bz(u1: u128, u2: u128) -> S160 {
    if u1 >= u2 {
        s160_from_u128_mag(u1 - u2, true)
    } else {
        s160_from_u128_mag(u2 - u1, false)
    }
}

#[inline]
fn u128_plus_u128_to_s160_mag(u1: u128, u2: u128) -> S160 {
    let u1_lo = u1 as u64;
    let u1_hi = (u1 >> 64) as u64;
    let u2_lo = u2 as u64;
    let u2_hi = (u2 >> 64) as u64;
    let (sum_lo, carry0) = u1_lo.overflowing_add(u2_lo);
    let (sum_hi1, carry1) = u1_hi.overflowing_add(u2_hi);
    let (sum_hi, carry2) = sum_hi1.overflowing_add(if carry0 { 1 } else { 0 });
    let carry_out = (carry1 as u8 | carry2 as u8) != 0;
    S160::new([sum_lo, sum_hi], if carry_out { 1 } else { 0 }, true)
}

#[inline]
fn u128_minus_i128_to_bz(u: u128, i: i128) -> S160 {
    if i >= 0 {
        u128_diff_to_bz(u, i as u128)
    } else {
        let abs_i = i.unsigned_abs();
        u128_plus_u128_to_s160_mag(u, abs_i)
    }
}

#[allow(unreachable_patterns)]
pub fn eval_az_by_name<F: JoltField>(
    c: &NamedConstraint,
    accessor: &dyn WitnessRowAccessor<F, JoltR1CSInputs>,
    row: usize,
) -> I8OrI96 {
    use JoltR1CSInputs as Inp;
    match c.name {
        ConstraintName::LeftInputEqRs1 => {
            // Az: LeftOperandIsRs1Value flag (0/1)
            accessor
                .value_at_bool(Inp::OpFlags(CircuitFlags::LeftOperandIsRs1Value), row)
                .into()
        }
        ConstraintName::LeftInputEqPC => {
            // Az: LeftOperandIsPC flag (0/1)
            accessor.value_at_bool(Inp::OpFlags(CircuitFlags::LeftOperandIsPC), row).into()
        }
        ConstraintName::RightInputEqRs2 => {
            // Az: RightOperandIsRs2Value flag (0/1)
            accessor.value_at_bool(Inp::OpFlags(CircuitFlags::RightOperandIsRs2Value), row).into()
        }
        ConstraintName::RightInputEqImm => {
            // Az: RightOperandIsImm flag (0/1)
            accessor.value_at_bool(Inp::OpFlags(CircuitFlags::RightOperandIsImm), row).into()
        }
        ConstraintName::RamReadEqRamWriteIfLoad => {
            // Az: Load flag (0/1)
            accessor.value_at_bool(Inp::OpFlags(CircuitFlags::Load), row).into()
        }
        ConstraintName::RamReadEqRdWriteIfLoad => {
            // Az: Load flag (0/1)
            accessor.value_at_bool(Inp::OpFlags(CircuitFlags::Load), row).into()
        }
        ConstraintName::LeftInputZeroOtherwise => {
            let f1 = accessor.value_at_bool(Inp::OpFlags(CircuitFlags::LeftOperandIsRs1Value), row);
            let f2 = accessor.value_at_bool(Inp::OpFlags(CircuitFlags::LeftOperandIsPC), row);
            // NOTE: relies on exclusivity of these circuit flags (validated in tests):
            // return 1 only if neither flag is set
            (!(f1 || f2)).into()
        }
        ConstraintName::RamAddrEqRs1PlusImmIfLoadStore => {
            // Az: Load OR Store flag (0/1)
            let load = accessor.value_at_bool(Inp::OpFlags(CircuitFlags::Load), row);
            let store = accessor.value_at_bool(Inp::OpFlags(CircuitFlags::Store), row);
            (load || store).into()
        }
        ConstraintName::LeftLookupZeroUnlessAddSubMul => {
            let add = accessor.value_at_bool(Inp::OpFlags(CircuitFlags::AddOperands), row);
            let sub = accessor.value_at_bool(Inp::OpFlags(CircuitFlags::SubtractOperands), row);
            let mul = accessor.value_at_bool(Inp::OpFlags(CircuitFlags::MultiplyOperands), row);
            // NOTE: these are exclusive circuit flags (validated in tests)
            (add || sub || mul).into()
        }
        ConstraintName::RightLookupAdd => {
            // Az: AddOperands flag (0/1)
            accessor.value_at_bool(Inp::OpFlags(CircuitFlags::AddOperands), row).into()
        }
        ConstraintName::RightLookupSub => {
            // Az: SubtractOperands flag (0/1)
            accessor.value_at_bool(Inp::OpFlags(CircuitFlags::SubtractOperands), row).into()
        }
        ConstraintName::ProductDef => {
            // Use unsigned left operand (bit pattern) to match Product witness convention
            I8OrI96::from(accessor.value_at_u64(Inp::LeftInstructionInput, row))
        }
        ConstraintName::RightLookupEqProductIfMul => {
            // Az: MultiplyOperands flag (0/1)
            accessor.value_at_bool(Inp::OpFlags(CircuitFlags::MultiplyOperands), row).into()
        }
        ConstraintName::RightLookupEqRightInputOtherwise => {
            let add = accessor.value_at_bool(Inp::OpFlags(CircuitFlags::AddOperands), row);
            let sub = accessor.value_at_bool(Inp::OpFlags(CircuitFlags::SubtractOperands), row);
            let mul = accessor.value_at_bool(Inp::OpFlags(CircuitFlags::MultiplyOperands), row);
            let adv = accessor.value_at_bool(Inp::OpFlags(CircuitFlags::Advice), row);
            // NOTE: relies on exclusivity of circuit flags (validated in tests):
            // return 1 only if none of add/sub/mul/adv is set
            (!(add || sub || mul || adv)).into()
        }
        ConstraintName::AssertLookupOne => {
            // Az: Assert flag (0/1)
            accessor.value_at_bool(Inp::OpFlags(CircuitFlags::Assert), row).into()
        }
        ConstraintName::WriteLookupOutputToRDDef => {
            // Az: Rd register index (0 disables write)
            I8OrI96::from_i8(accessor.value_at_u8(Inp::Rd, row) as i8)
        }
        ConstraintName::RdWriteEqLookupIfWriteLookupToRd => {
            // Az: WriteLookupOutputToRD indicator (0/1)
            I8OrI96::from_i8(accessor.value_at_u8(Inp::WriteLookupOutputToRD, row) as i8)
        }
        ConstraintName::WritePCtoRDDef => {
            // Az: Rd register index (0 disables write)
            I8OrI96::from_i8(accessor.value_at_u8(Inp::Rd, row) as i8)
        }
        ConstraintName::RdWriteEqPCPlusConstIfWritePCtoRD => {
            // Az: WritePCtoRD indicator (0/1)
            I8OrI96::from_i8(accessor.value_at_u8(Inp::WritePCtoRD, row) as i8)
        }
        ConstraintName::ShouldJumpDef => {
            // Az: Jump flag (0/1)
            accessor.value_at_bool(Inp::OpFlags(CircuitFlags::Jump), row).into()
        }
        ConstraintName::NextUnexpPCEqLookupIfShouldJump => {
            // Az: ShouldJump indicator (0/1)
            accessor.value_at_bool(Inp::ShouldJump, row).into()
        }
        ConstraintName::ShouldBranchDef => {
            // Az: Branch flag (0/1)
            accessor.value_at_bool(Inp::OpFlags(CircuitFlags::Branch), row).into()
        }
        ConstraintName::NextUnexpPCEqPCPlusImmIfShouldBranch => {
            // Az: ShouldBranch indicator (0/1)
            (accessor.value_at_u64(Inp::ShouldBranch, row) != 0).into()
        }
        ConstraintName::NextUnexpPCUpdateOtherwise => {
            // Az encodes 1 - ShouldBranch - Jump = (1 - Jump) - ShouldBranch.
            let should_branch_u64 = accessor.value_at_u64(Inp::ShouldBranch, row);
            let jump_flag = accessor.value_at_bool(Inp::OpFlags(CircuitFlags::Jump), row);
            let not_jump: i128 = if jump_flag { 0 } else { 1 };
            let diff = not_jump - (should_branch_u64 as i128);
            I8OrI96::from(diff)
        }
        ConstraintName::NextPCEqPCPlusOneIfInline => {
            // Az: InlineSequenceInstruction flag (0/1)
            accessor.value_at_bool(Inp::OpFlags(CircuitFlags::InlineSequenceInstruction), row).into()
        }
        ConstraintName::RightInputZeroOtherwise => {
            let f1 =
                accessor.value_at_bool(Inp::OpFlags(CircuitFlags::RightOperandIsRs2Value), row);
            let f2 = accessor.value_at_bool(Inp::OpFlags(CircuitFlags::RightOperandIsImm), row);
            // NOTE: relies on exclusivity of these circuit flags (validated in tests):
            // return 1 only if neither flag is set
            (!(f1 || f2)).into()
        }
        ConstraintName::Rs2EqRamWriteIfStore => {
            // Az: Store flag (0/1)
            accessor.value_at_bool(Inp::OpFlags(CircuitFlags::Store), row).into()
        }
    }
}

#[allow(unreachable_patterns)]
pub fn eval_bz_by_name<F: JoltField>(
    c: &NamedConstraint,
    accessor: &dyn WitnessRowAccessor<F, JoltR1CSInputs>,
    row: usize,
) -> S160 {
    use JoltR1CSInputs as Inp;
    match c.name {
        ConstraintName::LeftInputEqRs1 => {
            // B: LeftInstructionInput - Rs1Value (signed-magnitude over u64 bit patterns)
            let left = accessor.value_at_u64(Inp::LeftInstructionInput, row);
            let rs1 = accessor.value_at_u64(Inp::Rs1Value, row);
            s160_from_u64_diff(left, rs1)
        }
        ConstraintName::LeftInputEqPC => {
            // B: LeftInstructionInput - UnexpandedPC (signed-magnitude over u64 bit patterns)
            let left = accessor.value_at_u64(Inp::LeftInstructionInput, row);
            let pc = accessor.value_at_u64(Inp::UnexpandedPC, row);
            s160_from_u64_diff(left, pc)
        }
        ConstraintName::RightInputEqRs2 => {
            // B: RightInstructionInput - Rs2Value (i128 arithmetic)
            let right_i128 = accessor
                .value_at_s64(Inp::RightInstructionInput, row)
                .to_i128();
            let rs2_u64 = accessor.value_at_u64(Inp::Rs2Value, row);
            diff_to_bz(right_i128 - (rs2_u64 as i128))
        }
        ConstraintName::RightInputEqImm => {
            // B: RightInstructionInput - Imm (i128 arithmetic)
            let right_i128 = accessor
                .value_at_s64(Inp::RightInstructionInput, row)
                .to_i128();
            let imm_i128 = accessor.value_at_s64(Inp::Imm, row).to_i128();
            diff_to_bz(right_i128 - imm_i128)
        }
        ConstraintName::RamReadEqRamWriteIfLoad => {
            // B: RamReadValue - RamWriteValue (u64 bit-pattern difference)
            let rd = accessor.value_at_u64(Inp::RamReadValue, row);
            let wr = accessor.value_at_u64(Inp::RamWriteValue, row);
            s160_from_u64_diff(rd, wr)
        }
        ConstraintName::RamReadEqRdWriteIfLoad => {
            // B: RamReadValue - RdWriteValue (u64 bit-pattern difference)
            let rd = accessor.value_at_u64(Inp::RamReadValue, row);
            let rdw = accessor.value_at_u64(Inp::RdWriteValue, row);
            s160_from_u64_diff(rd, rdw)
        }
        ConstraintName::LeftInputZeroOtherwise => {
            // B: LeftInstructionInput - 0 (u64 bit pattern)
            let left = accessor.value_at_u64(Inp::LeftInstructionInput, row);
            s160_from_u64_diff(left, 0)
        }
        ConstraintName::RamAddrEqRs1PlusImmIfLoadStore => {
            // B: (Rs1Value + Imm) - 0 (true_val - false_val from if-else)
            let rs1 = accessor.value_at_u64(Inp::Rs1Value, row) as i128;
            let imm = accessor.value_at_s64(Inp::Imm, row).to_i128();
            diff_to_bz(rs1 + imm)
        }
        ConstraintName::LeftLookupZeroUnlessAddSubMul => {
            // B: 0 - LeftInstructionInput (true_val - false_val from if-else)
            let left = accessor.value_at_u64(Inp::LeftInstructionInput, row) as i128;
            diff_to_bz(0 - left)
        }
        ConstraintName::RightLookupAdd => {
            // B: RightLookupOperand - (LeftInstructionInput + RightInstructionInput) with full-width integer semantics
            let left_u64 = accessor.value_at_u64(Inp::LeftInstructionInput, row);
            let right_i128 = accessor
                .value_at_s64(Inp::RightInstructionInput, row)
                .to_i128();
            let r_u128 = accessor.value_at_u128(Inp::RightLookupOperand, row);
            let expected_i128 = (left_u64 as i128) + right_i128;
            u128_minus_i128_to_bz(r_u128, expected_i128)
        }
        ConstraintName::RightLookupSub => {
            // B: RightLookupOperand - (LeftInstructionInput - RightInstructionInput) with full-width integer semantics
            let left_u64 = accessor.value_at_u64(Inp::LeftInstructionInput, row);
            let right_i128 = accessor
                .value_at_s64(Inp::RightInstructionInput, row)
                .to_i128();
            let r_u128 = accessor.value_at_u128(Inp::RightLookupOperand, row);
            let expected = (left_u64 as i128) - right_i128;
            u128_minus_i128_to_bz(r_u128, expected)
        }
        ConstraintName::ProductDef => {
            // B: RightInstructionInput (exact signed value as i128)
            let right_i128 = accessor
                .value_at_s64(Inp::RightInstructionInput, row)
                .to_i128();
            S160::from(right_i128)
        }
        ConstraintName::RightLookupEqProductIfMul => {
            // B: RightLookupOperand - Product with full 128-bit semantics
            let a = accessor.value_at_u128(Inp::RightLookupOperand, row);
            let prod = accessor.value_at_s128(Inp::Product, row);
            if let Some(p_i128) = prod.to_i128() {
                u128_minus_i128_to_bz(a, p_i128)
            } else {
                let b = prod.magnitude_as_u128();
                if prod.sign() {
                    // a - |b|
                    u128_diff_to_bz(a, b)
                } else {
                    // a - (-|b|) = a + |b|
                    u128_plus_u128_to_s160_mag(a, b)
                }
            }
        }
        ConstraintName::RightLookupEqRightInputOtherwise => {
            // B: RightLookupOperand - RightInstructionInput with exact integer semantics
            let rlookup = accessor.value_at_u128(Inp::RightLookupOperand, row);
            let right = accessor
                .value_at_s64(Inp::RightInstructionInput, row)
                .to_i128();
            u128_minus_i128_to_bz(rlookup, right)
        }
        ConstraintName::AssertLookupOne => {
            // B: LookupOutput - 1 (i128 arithmetic)
            let lookup = accessor.value_at_u64(Inp::LookupOutput, row) as i128;
            diff_to_bz(lookup - 1)
        }
        ConstraintName::WriteLookupOutputToRDDef => {
            // B: OpFlags(WriteLookupOutputToRD) (boolean 0/1)
            let flag =
                accessor.value_at_bool(Inp::OpFlags(CircuitFlags::WriteLookupOutputToRD), row);
            diff_to_bz(if flag { 1 } else { 0 })
        }
        ConstraintName::RdWriteEqLookupIfWriteLookupToRd => {
            // B: RdWriteValue - LookupOutput (u64 bit-pattern difference)
            let rdw = accessor.value_at_u64(Inp::RdWriteValue, row);
            let lookup = accessor.value_at_u64(Inp::LookupOutput, row);
            s160_from_u64_diff(rdw, lookup)
        }
        ConstraintName::WritePCtoRDDef => {
            // B: OpFlags(Jump) (boolean 0/1)
            let jump = accessor.value_at_bool(Inp::OpFlags(CircuitFlags::Jump), row);
            diff_to_bz(if jump { 1 } else { 0 })
        }
        ConstraintName::RdWriteEqPCPlusConstIfWritePCtoRD => {
            // B: RdWriteValue - (UnexpandedPC + (4 - 2*IsCompressed))
            let rdw = accessor.value_at_u64(Inp::RdWriteValue, row) as i128;
            let pc = accessor.value_at_u64(Inp::UnexpandedPC, row) as i128;
            let is_compr = accessor.value_at_bool(Inp::OpFlags(CircuitFlags::IsCompressed), row);
            let const_term = 4 - if is_compr { 2 } else { 0 };
            diff_to_bz(rdw - (pc + const_term as i128))
        }
        ConstraintName::ShouldJumpDef => {
            // B: 1 - NextIsNoop (boolean domain)
            let next_noop = accessor.value_at_bool(Inp::NextIsNoop, row);
            diff_to_bz(if next_noop { 0 } else { 1 })
        }
        ConstraintName::NextUnexpPCEqLookupIfShouldJump => {
            // B: NextUnexpandedPC - LookupOutput (i128 arithmetic)
            let nextpc = accessor.value_at_u64(Inp::NextUnexpandedPC, row) as i128;
            let lookup = accessor.value_at_u64(Inp::LookupOutput, row) as i128;
            diff_to_bz(nextpc - lookup)
        }
        ConstraintName::ShouldBranchDef => {
            // B: LookupOutput (u64 bit pattern)
            let lookup = accessor.value_at_u64(Inp::LookupOutput, row);
            s160_from_u64_diff(lookup, 0)
        }
        ConstraintName::NextUnexpPCEqPCPlusImmIfShouldBranch => {
            // B: NextUnexpandedPC - (UnexpandedPC + Imm)
            let next = accessor.value_at_u64(Inp::NextUnexpandedPC, row) as i128;
            let pc = accessor.value_at_u64(Inp::UnexpandedPC, row) as i128;
            let imm = accessor.value_at_s64(Inp::Imm, row).to_i128();
            diff_to_bz(next - (pc + imm))
        }
        ConstraintName::NextUnexpPCUpdateOtherwise => {
            // B: NextUnexpandedPC - target, where target = UnexpandedPC + 4 - 4*DoNotUpdateUnexpandedPC - 2*IsCompressed
            let next = accessor.value_at_u64(Inp::NextUnexpandedPC, row) as i128;
            let pc = accessor.value_at_u64(Inp::UnexpandedPC, row) as i128;
            let dnoupd = accessor
                .value_at_bool(Inp::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC), row)
                as i128;
            let iscompr =
                accessor.value_at_bool(Inp::OpFlags(CircuitFlags::IsCompressed), row) as i128;
            let target = pc + 4 - 4 * dnoupd - 2 * iscompr;
            diff_to_bz(next - target)
        }
        ConstraintName::NextPCEqPCPlusOneIfInline => {
            // B: NextPC - (PC + 1)
            let next = accessor.value_at_u64(Inp::NextPC, row) as i128;
            let pc = accessor.value_at_u64(Inp::PC, row) as i128;
            diff_to_bz(next - (pc + 1))
        }
        ConstraintName::RightInputZeroOtherwise => {
            // B: RightInstructionInput - 0 (i128 arithmetic)
            let right = accessor
                .value_at_s64(Inp::RightInstructionInput, row)
                .to_i128();
            diff_to_bz(right)
        }
        ConstraintName::Rs2EqRamWriteIfStore => {
            // B: Rs2Value - RamWriteValue (u64 bit-pattern difference)
            let rs2 = accessor.value_at_u64(Inp::Rs2Value, row);
            let wr = accessor.value_at_u64(Inp::RamWriteValue, row);
            s160_from_u64_diff(rs2, wr)
        }
    }
}

// =============================================================================
// Batch evaluation functions
// =============================================================================

/// Batch evaluation of Az and Bz values for a chunk of constraints at a given step.
/// This is more efficient than evaluating constraints one by one as it reduces
/// function call overhead and enables better optimization.
#[inline]
pub fn eval_az_bz_batch<F: JoltField>(
    constraints: &[NamedConstraint],
    accessor: &dyn WitnessRowAccessor<F, JoltR1CSInputs>,
    step_idx: usize,
    az_output: &mut [I8OrI96],
    bz_output: &mut [S160],
) {
    debug_assert_eq!(constraints.len(), az_output.len());
    debug_assert_eq!(constraints.len(), bz_output.len());

    // Batch process all constraints for this step
    // This reduces virtual function call overhead compared to individual evaluations
    // Future optimization: could potentially share computations between constraints
    // that use similar input patterns, but for now we focus on reducing call overhead
    for (i, constraint) in constraints.iter().enumerate() {
        // Prefer named evaluators to preserve exact integer semantics (e.g., 2^64 constants)
        az_output[i] = eval_az_by_name(constraint, accessor, step_idx);
        bz_output[i] = eval_bz_by_name(constraint, accessor, step_idx);
    }
}
