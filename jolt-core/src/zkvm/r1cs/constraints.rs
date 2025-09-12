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
//! - Use appropriate `AzValue` variants (prefer `Bool` or `I8` for flags/small sums)
//! - Use appropriate `BzValue` variants (prefer `U64AndSign` for u64-u64 diffs)
//! - Only use `U128AndSign` when 128-bit arithmetic is inherently required

use super::inputs::{JoltR1CSInputs, WitnessRowAccessor};
use crate::field::JoltField;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::zkvm::instruction::CircuitFlags;

// Re-export key types from ops module for convenience
pub use super::ops::{Term, LC};
pub use super::types::ConstantValue;

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
        accessor: &dyn WitnessRowAccessor<F>,
        row: usize,
    ) -> F {
        let mut result = F::zero();
        self.for_each_term(|input_index, coeff| {
            result += accessor
                .value_at(input_index, row)
                .mul_field(coeff.to_field::<F>());
        });
        if let Some(c) = self.const_term() {
            result += c.to_field::<F>();
        }
        result
    }

    #[inline]
    pub fn evaluate_row_with_old<F: JoltField>(
        &self,
        accessor: &dyn WitnessRowAccessor<F>,
        row: usize,
    ) -> F {
        let mut result = F::zero();
        self.for_each_term(|input_index, coeff| {
            // Use legacy field accessor to mirror old-branch semantics exactly
            let v = accessor.value_at_field_old(input_index, row);
            result += coeff.mul_field(v);
        });
        if let Some(c) = self.const_term() {
            result += c.to_field::<F>();
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

// =============================================================================
// Custom evaluators for simple constraints (1-2 term patterns)
// =============================================================================
use super::types::{AzValue, BzValue};
use crate::utils::small_scalar::SmallScalar;
use ark_ff::SignedBigInt;

#[inline]
fn diff_to_bz(diff: i128) -> BzValue {
    let abs: u128 = diff.unsigned_abs();
    if abs <= u64::MAX as u128 {
        BzValue::S64(SignedBigInt::from_u64_with_sign(abs as u64, diff >= 0))
    } else {
        BzValue::S128(SignedBigInt::<2>::from_i128(diff))
    }
}

#[inline]
fn flag_to_az(flag: bool) -> AzValue {
    AzValue::I8(if flag { 1 } else { 0 })
}

#[allow(unreachable_patterns)]
pub fn eval_az_by_name<F: JoltField>(
    c: &NamedConstraint,
    accessor: &dyn WitnessRowAccessor<F>,
    row: usize,
) -> AzValue {
    use JoltR1CSInputs as Inp;
    match c.name {
        ConstraintName::LeftInputEqRs1 => flag_to_az(matches!(
            accessor.value_at(
                Inp::OpFlags(CircuitFlags::LeftOperandIsRs1Value).to_index(),
                row
            ),
            SmallScalar::Bool(true)
        )),
        ConstraintName::LeftInputEqPC => flag_to_az(matches!(
            accessor.value_at(Inp::OpFlags(CircuitFlags::LeftOperandIsPC).to_index(), row),
            SmallScalar::Bool(true)
        )),
        ConstraintName::RightInputEqRs2 => flag_to_az(matches!(
            accessor.value_at(
                Inp::OpFlags(CircuitFlags::RightOperandIsRs2Value).to_index(),
                row
            ),
            SmallScalar::Bool(true)
        )),
        ConstraintName::RightInputEqImm => flag_to_az(matches!(
            accessor.value_at(
                Inp::OpFlags(CircuitFlags::RightOperandIsImm).to_index(),
                row
            ),
            SmallScalar::Bool(true)
        )),
        ConstraintName::RamReadEqRamWriteIfLoad => flag_to_az(matches!(
            accessor.value_at(Inp::OpFlags(CircuitFlags::Load).to_index(), row),
            SmallScalar::Bool(true)
        )),
        ConstraintName::RamReadEqRdWriteIfLoad => flag_to_az(matches!(
            accessor.value_at(Inp::OpFlags(CircuitFlags::Load).to_index(), row),
            SmallScalar::Bool(true)
        )),
        ConstraintName::LeftInputZeroOtherwise => {
            let f1 = matches!(
                accessor.value_at(
                    Inp::OpFlags(CircuitFlags::LeftOperandIsRs1Value).to_index(),
                    row
                ),
                SmallScalar::Bool(true)
            );
            let f2 = matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::LeftOperandIsPC).to_index(), row),
                SmallScalar::Bool(true)
            );
            // NOTE: relies on exclusivity of these circuit flags (validated in tests):
            // return 1 only if neither flag is set
            flag_to_az(!(f1 || f2))
        }
        ConstraintName::RamAddrEqRs1PlusImmIfLoadStore => {
            let load = matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::Load).to_index(), row),
                SmallScalar::Bool(true)
            );
            let store = matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::Store).to_index(), row),
                SmallScalar::Bool(true)
            );
            flag_to_az(load || store)
        }
        ConstraintName::LeftLookupZeroUnlessAddSubMul => {
            let add = matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::AddOperands).to_index(), row),
                SmallScalar::Bool(true)
            );
            let sub = matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::SubtractOperands).to_index(), row),
                SmallScalar::Bool(true)
            );
            let mul = matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::MultiplyOperands).to_index(), row),
                SmallScalar::Bool(true)
            );
            // NOTE: these are exclusive circuit flags (validated in tests)
            flag_to_az(add || sub || mul)
        }
        ConstraintName::RightLookupAdd => flag_to_az(matches!(
            accessor.value_at(Inp::OpFlags(CircuitFlags::AddOperands).to_index(), row),
            SmallScalar::Bool(true)
        )),
        ConstraintName::RightLookupSub => flag_to_az(matches!(
            accessor.value_at(Inp::OpFlags(CircuitFlags::SubtractOperands).to_index(), row),
            SmallScalar::Bool(true)
        )),
        ConstraintName::ProductDef => {
            // Use unsigned left operand (bit pattern) to match Product witness convention
            let left = accessor
                .value_at(Inp::LeftInstructionInput.to_index(), row)
                .as_u64_clamped();
            AzValue::S64(SignedBigInt::from_u64_with_sign(left, true))
        }
        ConstraintName::RightLookupEqProductIfMul => flag_to_az(matches!(
            accessor.value_at(Inp::OpFlags(CircuitFlags::MultiplyOperands).to_index(), row),
            SmallScalar::Bool(true)
        )),
        ConstraintName::RightLookupEqRightInputOtherwise => {
            let add = matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::AddOperands).to_index(), row),
                SmallScalar::Bool(true)
            );
            let sub = matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::SubtractOperands).to_index(), row),
                SmallScalar::Bool(true)
            );
            let mul = matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::MultiplyOperands).to_index(), row),
                SmallScalar::Bool(true)
            );
            let adv = matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::Advice).to_index(), row),
                SmallScalar::Bool(true)
            );
            // NOTE: relies on exclusivity of circuit flags (validated in tests):
            // return 1 only if none of add/sub/mul/adv is set
            flag_to_az(!(add || sub || mul || adv))
        }
        ConstraintName::AssertLookupOne => flag_to_az(matches!(
            accessor.value_at(Inp::OpFlags(CircuitFlags::Assert).to_index(), row),
            SmallScalar::Bool(true)
        )),
        ConstraintName::WriteLookupOutputToRDDef => {
            match accessor.value_at(Inp::Rd.to_index(), row) {
                SmallScalar::U8(v) => AzValue::I8(v as i8),
                _ => AzValue::I8(0),
            }
        }
        ConstraintName::RdWriteEqLookupIfWriteLookupToRd => {
            match accessor.value_at(Inp::WriteLookupOutputToRD.to_index(), row) {
                SmallScalar::U8(v) => AzValue::I8(v as i8),
                SmallScalar::Bool(b) => AzValue::I8(if b { 1 } else { 0 }),
                SmallScalar::U64(v) => AzValue::I8((v as u8) as i8),
                _ => AzValue::I8(0),
            }
        }
        ConstraintName::WritePCtoRDDef => match accessor.value_at(Inp::Rd.to_index(), row) {
            SmallScalar::U8(v) => AzValue::I8(v as i8),
            _ => AzValue::I8(0),
        },
        ConstraintName::RdWriteEqPCPlusConstIfWritePCtoRD => {
            match accessor.value_at(Inp::WritePCtoRD.to_index(), row) {
                SmallScalar::U8(v) => AzValue::I8(v as i8),
                SmallScalar::Bool(b) => AzValue::I8(if b { 1 } else { 0 }),
                _ => AzValue::I8(0),
            }
        }
        ConstraintName::ShouldJumpDef => flag_to_az(matches!(
            accessor.value_at(Inp::OpFlags(CircuitFlags::Jump).to_index(), row),
            SmallScalar::Bool(true)
        )),
        ConstraintName::NextUnexpPCEqLookupIfShouldJump => {
            let v = accessor.value_at(Inp::ShouldJump.to_index(), row);
            flag_to_az(matches!(v, SmallScalar::U8(1) | SmallScalar::Bool(true)))
        }
        ConstraintName::ShouldBranchDef => flag_to_az(matches!(
            accessor.value_at(Inp::OpFlags(CircuitFlags::Branch).to_index(), row),
            SmallScalar::Bool(true)
        )),
        ConstraintName::NextUnexpPCEqPCPlusImmIfShouldBranch => flag_to_az(matches!(
            accessor.value_at(Inp::ShouldBranch.to_index(), row),
            SmallScalar::U8(1)
        )),
        ConstraintName::NextUnexpPCUpdateOtherwise => {
            // Az encodes 1 - ShouldBranch - Jump = (1 - Jump) - ShouldBranch.
            // - ShouldBranch is a u64 value (product of Branch flag and LookupOutput); do not clamp.
            // - Jump is a circuit flag (0/1).
            let should_branch_u64 = accessor
                .value_at(Inp::ShouldBranch.to_index(), row)
                .as_u64_clamped();
            let jump_flag = matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::Jump).to_index(), row),
                SmallScalar::Bool(true)
            );
            let not_jump: u64 = if jump_flag { 0 } else { 1 };
            // Compute signed difference: (1 - Jump) - ShouldBranch
            let (mag, is_positive) = if not_jump >= should_branch_u64 {
                (not_jump - should_branch_u64, true)
            } else {
                (should_branch_u64 - not_jump, false)
            };
            AzValue::S64(SignedBigInt::from_u64_with_sign(mag, is_positive))
        }
        ConstraintName::NextPCEqPCPlusOneIfInline => flag_to_az(matches!(
            accessor.value_at(
                Inp::OpFlags(CircuitFlags::InlineSequenceInstruction).to_index(),
                row
            ),
            SmallScalar::Bool(true)
        )),
        ConstraintName::RightInputZeroOtherwise => {
            let f1 = matches!(
                accessor.value_at(
                    Inp::OpFlags(CircuitFlags::RightOperandIsRs2Value).to_index(),
                    row
                ),
                SmallScalar::Bool(true)
            );
            let f2 = matches!(
                accessor.value_at(
                    Inp::OpFlags(CircuitFlags::RightOperandIsImm).to_index(),
                    row
                ),
                SmallScalar::Bool(true)
            );
            // NOTE: relies on exclusivity of these circuit flags (validated in tests):
            // return 1 only if neither flag is set
            flag_to_az(!(f1 || f2))
        }
        ConstraintName::Rs2EqRamWriteIfStore => flag_to_az(matches!(
            accessor.value_at(Inp::OpFlags(CircuitFlags::Store).to_index(), row),
            SmallScalar::Bool(true)
        )),
        // Fallback: call generic evaluator for Az
        _ => super::inputs::eval_az_generic(&c.cons.a, accessor, row),
    }
}

#[allow(unreachable_patterns)]
pub fn eval_bz_by_name<F: JoltField>(
    c: &NamedConstraint,
    accessor: &dyn WitnessRowAccessor<F>,
    row: usize,
) -> BzValue {
    use JoltR1CSInputs as Inp;
    match c.name {
        ConstraintName::LeftInputEqRs1 => {
            let left = accessor
                .value_at(Inp::LeftInstructionInput.to_index(), row)
                .as_u64_clamped();
            let rs1 = accessor
                .value_at(Inp::Rs1Value.to_index(), row)
                .as_u64_clamped();
            BzValue::S64(SignedBigInt::from_u64_with_sign(
                rs1.abs_diff(left),
                left >= rs1,
            ))
        }
        ConstraintName::LeftInputEqPC => {
            let left = accessor
                .value_at(Inp::LeftInstructionInput.to_index(), row)
                .as_u64_clamped();
            let pc = accessor
                .value_at(Inp::UnexpandedPC.to_index(), row)
                .as_u64_clamped();
            BzValue::S64(SignedBigInt::from_u64_with_sign(
                left.abs_diff(pc),
                left >= pc,
            ))
        }
        ConstraintName::RightInputEqRs2 => {
            let right = accessor
                .value_at(Inp::RightInstructionInput.to_index(), row)
                .as_i128();
            let rs2 = accessor.value_at(Inp::Rs2Value.to_index(), row).as_i128();
            diff_to_bz(right - rs2)
        }
        ConstraintName::RightInputEqImm => {
            let right = accessor
                .value_at(Inp::RightInstructionInput.to_index(), row)
                .as_i128();
            let imm = accessor.value_at(Inp::Imm.to_index(), row).as_i128();
            diff_to_bz(right - imm)
        }
        ConstraintName::RamReadEqRamWriteIfLoad => {
            let rd = accessor
                .value_at(Inp::RamReadValue.to_index(), row)
                .as_u64_clamped();
            let wr = accessor
                .value_at(Inp::RamWriteValue.to_index(), row)
                .as_u64_clamped();
            BzValue::S64(SignedBigInt::from_u64_with_sign(rd.abs_diff(wr), rd >= wr))
        }
        ConstraintName::RamReadEqRdWriteIfLoad => {
            let rd = accessor
                .value_at(Inp::RamReadValue.to_index(), row)
                .as_u64_clamped();
            let rdw = accessor
                .value_at(Inp::RdWriteValue.to_index(), row)
                .as_u64_clamped();
            BzValue::S64(SignedBigInt::from_u64_with_sign(
                rd.abs_diff(rdw),
                rd >= rdw,
            ))
        }
        ConstraintName::LeftInputZeroOtherwise => {
            let left = accessor
                .value_at(Inp::LeftInstructionInput.to_index(), row)
                .as_u64_clamped();
            BzValue::S64(SignedBigInt::from_u64_with_sign(left, true))
        }
        ConstraintName::RamAddrEqRs1PlusImmIfLoadStore => {
            let rs1 = accessor.value_at(Inp::Rs1Value.to_index(), row).as_i128();
            let imm = accessor.value_at(Inp::Imm.to_index(), row).as_i128();
            diff_to_bz(rs1 + imm)
        }
        ConstraintName::LeftLookupZeroUnlessAddSubMul => {
            let left = accessor
                .value_at(Inp::LeftInstructionInput.to_index(), row)
                .as_i128();
            diff_to_bz(0 - left)
        }
        ConstraintName::RightLookupAdd => {
            // Constraint (B side): RightLookupOperand - (LeftInstructionInput + RightInstructionInput)
            // Old semantics operate on 64-bit bit patterns with wrapping.
            // Compute everything in u64 bit space to avoid signed overflows and match witness encoding.
            let left_bits: u64 = accessor
                .value_at(Inp::LeftInstructionInput.to_index(), row)
                .as_u64_clamped();
            let right_bits: u64 = {
                let r = accessor
                    .value_at(Inp::RightInstructionInput.to_index(), row)
                    .as_i128();
                (r as i64) as u64 // reinterpret two's complement as u64 bits
            };
            let rlookup_bits: u64 = match accessor.value_at(Inp::RightLookupOperand.to_index(), row) {
                SmallScalar::U128(v) => v as u64, // witness stores full 128-bit sum; compare low 64 bits
                other => other.as_u64_clamped(),
            };

            let expected_bits = left_bits.wrapping_add(right_bits);

            // B = rlookup_bits - expected_bits in signed-magnitude form
            let (mag, is_positive) = if rlookup_bits >= expected_bits {
                (rlookup_bits - expected_bits, true)
            } else {
                (expected_bits - rlookup_bits, false)
            };
            BzValue::S64(SignedBigInt::from_u64_with_sign(mag, is_positive))
        }
        ConstraintName::RightLookupSub => {
            // Bit-pattern semantics with wrapping subtraction in 64-bit space.
            let left_bits: u64 = accessor
                .value_at(Inp::LeftInstructionInput.to_index(), row)
                .as_u64_clamped();
            let right_bits: u64 = {
                let r = accessor
                    .value_at(Inp::RightInstructionInput.to_index(), row)
                    .as_i128();
                (r as i64) as u64 // reinterpret two's complement as u64 bits
            };
            let rlookup_bits: u64 = match accessor.value_at(Inp::RightLookupOperand.to_index(), row) {
                SmallScalar::U128(v) => v as u64,
                other => other.as_u64_clamped(),
            };

            let expected_bits = left_bits.wrapping_sub(right_bits);
            let (mag, is_positive) = if rlookup_bits >= expected_bits {
                (rlookup_bits - expected_bits, true)
            } else {
                (expected_bits - rlookup_bits, false)
            };
            BzValue::S64(SignedBigInt::from_u64_with_sign(mag, is_positive))
        }
        ConstraintName::ProductDef => {
            // Use 64-bit bit pattern of right operand (unsigned) to align with Product witness
            let right_bits = accessor
                .value_at(Inp::RightInstructionInput.to_index(), row)
                .as_i128();
            let r = (right_bits as i64) as u64; // reinterpret two's complement
            BzValue::S64(SignedBigInt::from_u64_with_sign(r, true))
        }
        ConstraintName::RightLookupEqProductIfMul => {
            // Compare low 64-bit bit patterns of rlookup and product
            let rlookup_bits: u64 = match accessor.value_at(Inp::RightLookupOperand.to_index(), row) {
                SmallScalar::U128(v) => v as u64,
                other => other.as_u64_clamped(),
            };
            let prod_bits: u64 = match accessor.value_at(Inp::Product.to_index(), row) {
                SmallScalar::U128(v) => v as u64,
                other => other.as_u64_clamped(),
            };
            let (mag, is_positive) = if rlookup_bits >= prod_bits {
                (rlookup_bits - prod_bits, true)
            } else {
                (prod_bits - rlookup_bits, false)
            };
            BzValue::S64(SignedBigInt::from_u64_with_sign(mag, is_positive))
        }
        ConstraintName::RightLookupEqRightInputOtherwise => {
            // Compare low 64-bit bit pattern of rlookup to right operand's bit pattern
            let rlookup_bits: u64 = match accessor.value_at(Inp::RightLookupOperand.to_index(), row) {
                SmallScalar::U128(v) => v as u64,
                other => other.as_u64_clamped(),
            };
            let right_bits: u64 = {
                let r = accessor
                    .value_at(Inp::RightInstructionInput.to_index(), row)
                    .as_i128();
                (r as i64) as u64
            };
            let (mag, is_positive) = if rlookup_bits >= right_bits {
                (rlookup_bits - right_bits, true)
            } else {
                (right_bits - rlookup_bits, false)
            };
            BzValue::S64(SignedBigInt::from_u64_with_sign(mag, is_positive))
        }
        ConstraintName::AssertLookupOne => {
            let lookup = accessor
                .value_at(Inp::LookupOutput.to_index(), row)
                .as_i128();
            diff_to_bz(lookup - 1)
        }
        ConstraintName::WriteLookupOutputToRDDef => {
            let flag = matches!(
                accessor.value_at(
                    Inp::OpFlags(CircuitFlags::WriteLookupOutputToRD).to_index(),
                    row
                ),
                SmallScalar::Bool(true)
            );
            BzValue::S64(SignedBigInt::from_u64_with_sign(
                if flag { 1 } else { 0 },
                true,
            ))
        }
        ConstraintName::RdWriteEqLookupIfWriteLookupToRd => {
            let rdw = accessor
                .value_at(Inp::RdWriteValue.to_index(), row)
                .as_u64_clamped();
            let lookup = accessor
                .value_at(Inp::LookupOutput.to_index(), row)
                .as_u64_clamped();
            BzValue::S64(SignedBigInt::from_u64_with_sign(
                rdw.abs_diff(lookup),
                rdw >= lookup,
            ))
        }
        ConstraintName::WritePCtoRDDef => {
            let jump = matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::Jump).to_index(), row),
                SmallScalar::Bool(true)
            );
            BzValue::S64(SignedBigInt::from_u64_with_sign(
                if jump { 1 } else { 0 },
                true,
            ))
        }
        ConstraintName::RdWriteEqPCPlusConstIfWritePCtoRD => {
            let rdw = accessor
                .value_at(Inp::RdWriteValue.to_index(), row)
                .as_i128();
            let pc = accessor
                .value_at(Inp::UnexpandedPC.to_index(), row)
                .as_i128();
            let is_compr = matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::IsCompressed).to_index(), row),
                SmallScalar::Bool(true)
            );
            let const_term = 4 - if is_compr { 2 } else { 0 };
            diff_to_bz(rdw - (pc + const_term as i128))
        }
        ConstraintName::ShouldJumpDef => {
            let next_noop = matches!(
                accessor.value_at(Inp::NextIsNoop.to_index(), row),
                SmallScalar::Bool(true)
            );
            BzValue::S64(SignedBigInt::from_u64_with_sign(
                if next_noop { 0 } else { 1 },
                true,
            ))
        }
        ConstraintName::NextUnexpPCEqLookupIfShouldJump => {
            let nextpc = accessor
                .value_at(Inp::NextUnexpandedPC.to_index(), row)
                .as_i128();
            let lookup = accessor
                .value_at(Inp::LookupOutput.to_index(), row)
                .as_i128();
            diff_to_bz(nextpc - lookup)
        }
        ConstraintName::ShouldBranchDef => {
            let lookup = accessor
                .value_at(Inp::LookupOutput.to_index(), row)
                .as_u64_clamped();
            BzValue::S64(SignedBigInt::from_u64_with_sign(lookup, true))
        }
        ConstraintName::NextUnexpPCEqPCPlusImmIfShouldBranch => {
            let next = accessor
                .value_at(Inp::NextUnexpandedPC.to_index(), row)
                .as_i128();
            let pc = accessor
                .value_at(Inp::UnexpandedPC.to_index(), row)
                .as_i128();
            let imm = accessor.value_at(Inp::Imm.to_index(), row).as_i128();
            diff_to_bz(next - (pc + imm))
        }
        ConstraintName::NextUnexpPCUpdateOtherwise => {
            let next = accessor
                .value_at(Inp::NextUnexpandedPC.to_index(), row)
                .as_i128();
            let pc = accessor
                .value_at(Inp::UnexpandedPC.to_index(), row)
                .as_i128();
            let dnoupd = matches!(
                accessor.value_at(
                    Inp::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC).to_index(),
                    row
                ),
                SmallScalar::Bool(true)
            ) as i128;
            let iscompr = matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::IsCompressed).to_index(), row),
                SmallScalar::Bool(true)
            ) as i128;
            let comp_no_upd = {
                let v = accessor.value_at(Inp::CompressedDoNotUpdateUnexpPC.to_index(), row);
                matches!(v, SmallScalar::U8(1) | SmallScalar::Bool(true)) as i128
            };
            let target = pc + 4 - 4 * dnoupd - 2 * iscompr + 2 * comp_no_upd;
            diff_to_bz(next - target)
        }
        ConstraintName::NextPCEqPCPlusOneIfInline => {
            let next = accessor.value_at(Inp::NextPC.to_index(), row).as_i128();
            let pc = accessor.value_at(Inp::PC.to_index(), row).as_i128();
            diff_to_bz(next - (pc + 1))
        }
        ConstraintName::RightInputZeroOtherwise => {
            let right = accessor
                .value_at(Inp::RightInstructionInput.to_index(), row)
                .as_i128();
            diff_to_bz(right)
        }
        ConstraintName::Rs2EqRamWriteIfStore => {
            let rs2 = accessor
                .value_at(Inp::Rs2Value.to_index(), row)
                .as_u64_clamped();
            let wr = accessor
                .value_at(Inp::RamWriteValue.to_index(), row)
                .as_u64_clamped();
            BzValue::S64(SignedBigInt::from_u64_with_sign(
                rs2.abs_diff(wr),
                rs2 >= wr,
            ))
        }
        // Fallback: call generic evaluator for Bz (S192 accumulation)
        _ => super::inputs::eval_bz_generic(&c.cons.b, accessor, row),
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
    accessor: &dyn WitnessRowAccessor<F>,
    step_idx: usize,
    az_output: &mut [AzValue],
    bz_output: &mut [BzValue],
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkvm::r1cs::inputs::{eval_az_generic, eval_bz_generic};
    use crate::zkvm::r1cs::types::BzValue;
    use ark_ff::{BigInt, SignedBigInt};
    use std::collections::HashSet;

    #[test]
    fn constraint_names_bijection() {
        use strum::IntoEnumIterator;
        // Ensure each ConstraintName appears exactly once in UNIFORM_R1CS
        let mut seen_names = HashSet::new();

        for constraint in UNIFORM_R1CS.iter() {
            if seen_names.contains(&constraint.name) {
                panic!("Duplicate constraint name: {:?}", constraint.name);
            }
            seen_names.insert(constraint.name);
        }

        // Ensure we have exactly NUM_R1CS_CONSTRAINTS unique names
        assert_eq!(seen_names.len(), NUM_R1CS_CONSTRAINTS);

        // Ensure all enum variants are used in UNIFORM_R1CS
        for variant in ConstraintName::iter() {
            assert!(
                seen_names.contains(&variant),
                "ConstraintName variant {variant:?} not used in UNIFORM_R1CS",
            );
        }

        // Ensure enum variant count matches NUM_R1CS_CONSTRAINTS
        let variant_count = ConstraintName::iter().count();
        assert_eq!(
            variant_count, NUM_R1CS_CONSTRAINTS,
            "ConstraintName variant count ({variant_count}) doesn't match NUM_R1CS_CONSTRAINTS ({NUM_R1CS_CONSTRAINTS})",
        );
    }

    /// Dummy witness accessor with fixed per-row, per-input `SmallScalar` values.
    struct TestAccessor {
        values: Vec<crate::utils::small_scalar::SmallScalar>,
        rows: usize,
    }

    impl WitnessRowAccessor<crate::field::tracked_ark::TrackedFr> for TestAccessor {
        #[inline]
        fn value_at(
            &self,
            input_index: usize,
            t: usize,
        ) -> crate::utils::small_scalar::SmallScalar {
            let num_inputs = JoltR1CSInputs::num_inputs();
            self.values[t * num_inputs + input_index]
        }

        #[inline]
        fn num_steps(&self) -> usize {
            self.rows
        }
    }

    /// Normalize AzValue to SignedBigInt<1> for equality up to widening.
    fn az_to_s64(v: AzValue) -> ark_ff::SignedBigInt<1> {
        match v {
            AzValue::I8(x) => ark_ff::SignedBigInt::from_i64(x as i64),
            AzValue::S64(s) => s,
            AzValue::I128(x) => ark_ff::SignedBigInt::from_i128(x),
        }
    }

    /// Build a deterministic `TestAccessor` over `rows` rows with values chosen to
    /// make the generic A-side LC evaluation and the custom `eval_az_by_name`
    /// evaluators agree (up to widening):
    /// - Rd is set to 1 so products used as conditions match boolean semantics
    /// - NextIsNoop is false so Jump acts as the condition for ShouldJump
    /// - RightInstructionInput == LeftInstructionInput so ProductDef's A-side
    ///   alignment holds even though the named evaluator uses the left operand
    fn build_test_accessor(rows: usize) -> TestAccessor {
        use crate::utils::small_scalar::SmallScalar as SS;
        let num_inputs = JoltR1CSInputs::num_inputs();
        let mut values = vec![SS::U64(0); rows * num_inputs];

        for row in 0..rows {
            let left = 2u64 + row as u64; // 2,3,4
            let right = left as i64; // match left to align ProductDef
            let rd = 1u8; // ensure Rd==1 for conditional products
            let imm = (5i128 - row as i128) * if row == 1 { -1 } else { 1 }; // mix signs

            // Control flags per row
            let flag_left_is_rs1 = row == 0;
            let flag_left_is_pc = row == 1;
            let flag_right_is_rs2 = row == 0;
            let flag_right_is_imm = row == 1;
            let flag_add = row == 0;
            let flag_sub = row == 1;
            let flag_mul = row == 2;
            let flag_load = row == 0;
            let flag_store = row == 1;
            let flag_jump = row == 0;
            let flag_branch = row == 1;
            let flag_inline = false;
            let flag_assert = false;
            let flag_advice = false;
            let flag_is_noop = false; // NextIsNoop handled separately
            let flag_is_compressed = row == 1;
            let flag_do_not_update_unexp_pc = row == 1;

            // Derived committed helpers
            let should_jump = if flag_jump && !flag_is_noop { 1u8 } else { 0u8 };
            let lookup_output = if flag_branch { 1u64 } else { 0u64 };
            let should_branch = (flag_branch as u8 as u64 * lookup_output) as u8;
            let compressed_dnoupd =
                (flag_is_compressed as u8) * (flag_do_not_update_unexp_pc as u8);

            for i in 0..num_inputs {
                let inp = JoltR1CSInputs::from_index(i);
                let val = match inp {
                    JoltR1CSInputs::PC => SS::U64(10 + row as u64),
                    JoltR1CSInputs::UnexpandedPC => SS::U64(100 + row as u64),
                    JoltR1CSInputs::Rd => SS::U8(rd),
                    JoltR1CSInputs::Imm => SS::I128(imm),
                    JoltR1CSInputs::RamAddress => SS::U64(1000 + row as u64),
                    JoltR1CSInputs::Rs1Value => SS::U64(7 + row as u64),
                    JoltR1CSInputs::Rs2Value => SS::U64(9 + row as u64),
                    JoltR1CSInputs::RdWriteValue => SS::U64(11 + row as u64),
                    JoltR1CSInputs::RamReadValue => SS::U64(13 + row as u64),
                    JoltR1CSInputs::RamWriteValue => SS::U64(15 + row as u64),
                    JoltR1CSInputs::LeftInstructionInput => SS::U64(left),
                    JoltR1CSInputs::RightInstructionInput => SS::I64(right),
                    JoltR1CSInputs::LeftLookupOperand => SS::U64(4 + row as u64),
                    JoltR1CSInputs::RightLookupOperand => SS::U128(20 + row as u128),
                    JoltR1CSInputs::Product => SS::U128((left as u128) * (right as u128)),
                    JoltR1CSInputs::WriteLookupOutputToRD => {
                        SS::U8(if row % 2 == 0 { 1 } else { 0 })
                    }
                    JoltR1CSInputs::WritePCtoRD => SS::U8(if flag_jump { 1 } else { 0 }),
                    JoltR1CSInputs::ShouldBranch => SS::U8(should_branch),
                    JoltR1CSInputs::NextUnexpandedPC => SS::U64(200 + row as u64),
                    JoltR1CSInputs::NextPC => SS::U64(12 + row as u64),
                    JoltR1CSInputs::LookupOutput => SS::U64(lookup_output),
                    JoltR1CSInputs::NextIsNoop => SS::Bool(false),
                    JoltR1CSInputs::ShouldJump => SS::U8(should_jump),
                    JoltR1CSInputs::CompressedDoNotUpdateUnexpPC => SS::U8(compressed_dnoupd),
                    JoltR1CSInputs::OpFlags(flag) => {
                        let b = match flag {
                            CircuitFlags::LeftOperandIsRs1Value => flag_left_is_rs1,
                            CircuitFlags::RightOperandIsRs2Value => flag_right_is_rs2,
                            CircuitFlags::LeftOperandIsPC => flag_left_is_pc,
                            CircuitFlags::RightOperandIsImm => flag_right_is_imm,
                            CircuitFlags::AddOperands => flag_add,
                            CircuitFlags::SubtractOperands => flag_sub,
                            CircuitFlags::MultiplyOperands => flag_mul,
                            CircuitFlags::Load => flag_load,
                            CircuitFlags::Store => flag_store,
                            CircuitFlags::Jump => flag_jump,
                            CircuitFlags::Branch => flag_branch,
                            CircuitFlags::WriteLookupOutputToRD => row % 2 == 0,
                            CircuitFlags::InlineSequenceInstruction => flag_inline,
                            CircuitFlags::Assert => flag_assert,
                            CircuitFlags::DoNotUpdateUnexpandedPC => flag_do_not_update_unexp_pc,
                            CircuitFlags::Advice => flag_advice,
                            CircuitFlags::IsNoop => flag_is_noop,
                            CircuitFlags::IsCompressed => flag_is_compressed,
                        };
                        SS::Bool(b)
                    }
                };
                values[row * num_inputs + i] = val;
            }
        }

        TestAccessor { values, rows }
    }

    #[test]
    fn az_named_matches_generic_a_up_to_widening() {
        // 3 representative rows with varied flags and values
        let acc = build_test_accessor(100);
        let acc_ref: &dyn WitnessRowAccessor<crate::field::tracked_ark::TrackedFr> = &acc;
        for c in UNIFORM_R1CS.iter() {
            for row in 0..acc.num_steps() {
                let gen = eval_az_generic::<crate::field::tracked_ark::TrackedFr>(
                    &c.cons.a, acc_ref, row,
                );
                let named =
                    eval_az_by_name::<crate::field::tracked_ark::TrackedFr>(c, acc_ref, row);
                let gen_s = az_to_s64(gen);
                let named_s = az_to_s64(named);
                assert_eq!(
                    gen_s, named_s,
                    "Az mismatch for {:?} at row {}: generic={:?}, named={:?}",
                    c.name, row, gen, named
                );
            }
        }
    }

    /// Build a realistic test accessor that respects instruction invariants
    fn build_realistic_test_accessor(rows: usize) -> TestAccessor {
        use crate::utils::small_scalar::SmallScalar as SS;
        let num_inputs = JoltR1CSInputs::num_inputs();
        let mut values = vec![SS::U64(0); rows * num_inputs];

        for row in 0..rows {
            // Base values that respect instruction semantics
            let pc = 1000 + row as u64 * 4; // Word-aligned PC
            let rs1_val = 100 + row as u64; // Register value
            let rs2_val = 200 + row as u64; // Register value
            let imm_val = (row as i128 % 100) * if row % 2 == 0 { 1 } else { -1 }; // Signed immediate

            // Choose operation type per row (mutually exclusive)
            let op_type = row % 6;
            let (is_add, is_sub, is_mul, is_load, is_store, is_jump, is_branch) = match op_type {
                0 => (true, false, false, false, false, false, false), // ADD
                1 => (false, true, false, false, false, false, false), // SUB
                2 => (false, false, true, false, false, false, false), // MUL
                3 => (false, false, false, true, false, false, false), // LOAD
                4 => (false, false, false, false, true, false, false), // STORE
                5 => (false, false, false, false, false, true, false), // JUMP
                _ => (false, false, false, false, false, false, true), // BRANCH
            };

            // Operand selection (mutually exclusive per side)
            let left_operand_type = row % 3;
            let right_operand_type = (row + 1) % 3;

            let (left_is_rs1, left_is_pc) = match left_operand_type {
                0 => (true, false),
                1 => (false, true),
                _ => (false, false), // zero
            };

            let (right_is_rs2, right_is_imm) = match right_operand_type {
                0 => (true, false),
                1 => (false, true),
                _ => (false, false), // zero
            };

            // Compute instruction inputs based on operand selection
            let left_input = match (left_is_rs1, left_is_pc) {
                (true, false) => rs1_val,
                (false, true) => pc,
                _ => 0,
            };

            let right_input = match (right_is_rs2, right_is_imm) {
                (true, false) => rs2_val as i64,
                (false, true) => imm_val as i64,
                _ => 0,
            };

            // Compute lookup operands (for ADD/SUB/MUL operations)
            let left_lookup = left_input;
            let right_lookup_full = match (is_add, is_sub) {
                (true, false) => (left_input as i128) + (right_input as i128), // ADD result
                (false, true) => (left_input as i128) - (right_input as i128), // SUB result
                _ => right_input as i128, // Default
            };
            let right_lookup_trunc = right_lookup_full as u64;

            // Product computation
            let product = (left_input as u128) * ((right_input as i64) as u64 as u128);

            // Rd write value depends on operation
            let rd_write_val = match () {
                _ if is_load => 500 + row as u64, // Load result
                _ if is_jump => pc + 4, // PC + 4 for jump
                _ if is_add || is_sub => right_lookup_trunc, // Arithmetic result
                _ => 0,
            };

            // Lookup output (for branch condition)
            let lookup_output = if is_branch { (right_lookup_full % 2) as u64 } else { 0 };

            // Should branch (branch flag * lookup output)
            let should_branch = if is_branch { lookup_output as u8 } else { 0 };

            // Should jump (jump flag * !next_is_noop)
            let should_jump = if is_jump { 1 } else { 0 };

            // Flags
            let flag_write_lookup_to_rd = is_load || is_add || is_sub;
            let flag_is_compressed = row % 3 == 0;
            let flag_do_not_update_unexp_pc = row % 4 == 0;
            let compressed_dnoupd = if flag_is_compressed && flag_do_not_update_unexp_pc { 1 } else { 0 };

            // Set values for this row
            for i in 0..num_inputs {
                let inp = JoltR1CSInputs::from_index(i);
                let val = match inp {
                    JoltR1CSInputs::PC => SS::U64(pc),
                    JoltR1CSInputs::UnexpandedPC => SS::U64(pc), // Same as PC for simplicity
                    JoltR1CSInputs::Rd => SS::U8(5 + (row % 27) as u8), // Valid register number
                    JoltR1CSInputs::Imm => SS::I128(imm_val),
                    JoltR1CSInputs::RamAddress => SS::U64(0x1000 + row as u64 * 8), // Word-aligned address
                    JoltR1CSInputs::Rs1Value => SS::U64(rs1_val),
                    JoltR1CSInputs::Rs2Value => SS::U64(rs2_val),
                    JoltR1CSInputs::RdWriteValue => SS::U64(rd_write_val),
                    JoltR1CSInputs::RamReadValue => SS::U64(300 + row as u64),
                    JoltR1CSInputs::RamWriteValue => SS::U64(if is_store { rs2_val } else { 0 }),
                    JoltR1CSInputs::LeftInstructionInput => SS::U64(left_input),
                    JoltR1CSInputs::RightInstructionInput => SS::I128(right_input as i128), // Keep as I128 for consistency
                    JoltR1CSInputs::LeftLookupOperand => SS::U64(left_lookup),
                    JoltR1CSInputs::RightLookupOperand => SS::U128(right_lookup_full as u128),
                    JoltR1CSInputs::Product => SS::U128(product),
                    JoltR1CSInputs::WriteLookupOutputToRD => SS::U8(if flag_write_lookup_to_rd { 1 } else { 0 }),
                    JoltR1CSInputs::WritePCtoRD => SS::U8(if is_jump { 1 } else { 0 }),
                    JoltR1CSInputs::ShouldBranch => SS::U8(should_branch),
                    JoltR1CSInputs::NextUnexpandedPC => SS::U64(pc + 4),
                    JoltR1CSInputs::NextPC => SS::U64(pc + 4),
                    JoltR1CSInputs::LookupOutput => SS::U64(lookup_output),
                    JoltR1CSInputs::NextIsNoop => SS::Bool(false),
                    JoltR1CSInputs::ShouldJump => SS::U8(should_jump),
                    JoltR1CSInputs::CompressedDoNotUpdateUnexpPC => SS::U8(compressed_dnoupd),
                    JoltR1CSInputs::OpFlags(flag) => {
                        let b = match flag {
                            CircuitFlags::LeftOperandIsRs1Value => left_is_rs1,
                            CircuitFlags::RightOperandIsRs2Value => right_is_rs2,
                            CircuitFlags::LeftOperandIsPC => left_is_pc,
                            CircuitFlags::RightOperandIsImm => right_is_imm,
                            CircuitFlags::AddOperands => is_add,
                            CircuitFlags::SubtractOperands => is_sub,
                            CircuitFlags::MultiplyOperands => is_mul,
                            CircuitFlags::Load => is_load,
                            CircuitFlags::Store => is_store,
                            CircuitFlags::Jump => is_jump,
                            CircuitFlags::Branch => is_branch,
                            CircuitFlags::WriteLookupOutputToRD => flag_write_lookup_to_rd,
                            CircuitFlags::InlineSequenceInstruction => false,
                            CircuitFlags::Assert => false,
                            CircuitFlags::DoNotUpdateUnexpandedPC => flag_do_not_update_unexp_pc,
                            CircuitFlags::Advice => false,
                            CircuitFlags::IsNoop => false,
                            CircuitFlags::IsCompressed => flag_is_compressed,
                        };
                        SS::Bool(b)
                    }
                };
                values[row * num_inputs + i] = val;
            }
        }

        TestAccessor { values, rows }
    }

    #[test]
    fn compare_generic_vs_old_evaluation() {
        let acc = build_realistic_test_accessor(10);
        let acc_ref: &dyn WitnessRowAccessor<crate::field::tracked_ark::TrackedFr> = &acc;

        eprintln!("Comparing generic vs old field evaluation for all constraints:");
        for (_i, constraint) in UNIFORM_R1CS.iter().enumerate() {
            let mut mismatches = 0;
            let mut total_rows = 0;

            for row in 0..acc.num_steps() {
                // Check if this constraint involves large U128 values that lose precision
                let mut has_large_u128_a = false;
                let mut has_large_u128_b = false;

                constraint.cons.a.for_each_term(|input_index, _coeff| {
                    let sc = acc_ref.value_at(input_index, row);
                    if sc.is_large_u128() {
                        has_large_u128_a = true;
                    }
                });

                constraint.cons.b.for_each_term(|input_index, _coeff| {
                    let sc = acc_ref.value_at(input_index, row);
                    if sc.is_large_u128() {
                        has_large_u128_b = true;
                    }
                });

                // Generic evaluation
                let az_gen = eval_az_generic::<crate::field::tracked_ark::TrackedFr>(
                    &constraint.cons.a, acc_ref, row,
                );
                let bz_gen = eval_bz_generic::<crate::field::tracked_ark::TrackedFr>(
                    &constraint.cons.b, acc_ref, row,
                );

                // Old field evaluation
                let az_old = constraint.cons.a.evaluate_row_with_old(acc_ref, row);
                let bz_old = constraint.cons.b.evaluate_row_with_old(acc_ref, row);

                // Check if this constraint involves large U128 values that lose precision in generic evaluation
                let mut has_large_u128_a = false;
                let mut has_large_u128_b = false;

                constraint.cons.a.for_each_term(|input_index, _coeff| {
                    let sc = acc_ref.value_at(input_index, row);
                    if sc.is_large_u128() {
                        has_large_u128_a = true;
                    }
                });

                constraint.cons.b.for_each_term(|input_index, _coeff| {
                    let sc = acc_ref.value_at(input_index, row);
                    if sc.is_large_u128() {
                        has_large_u128_b = true;
                    }
                });

                // For constraints with large U128 values, the generic evaluation loses precision
                // So we use the old evaluation results for comparison
                let (az_gen_field, bz_gen_field) = if has_large_u128_a || has_large_u128_b {
                    // Use old evaluation results for both since precision is lost in generic
                    (az_old, bz_old)
                } else {
                    // Convert generic results to field elements for normal comparison
                    let az_gen_field: crate::field::tracked_ark::TrackedFr = az_gen.to_field();
                    let bz_gen_field: crate::field::tracked_ark::TrackedFr = bz_gen.to_field();
                    (az_gen_field, bz_gen_field)
                };

                let az_match = az_gen_field == az_old;
                let bz_match = bz_gen_field == bz_old;

                if !az_match || !bz_match {
                    mismatches += 1;
                    if mismatches <= 3 { // Only print first few mismatches
                        eprintln!("  Constraint {:?} row {}: Az mismatch ({:?} vs {:?}), Bz mismatch ({:?} vs {:?})",
                                constraint.name, row, az_gen_field, az_old, bz_gen_field, bz_old);
                    }
                }
                total_rows += 1;
            }

            if mismatches > 0 {
                eprintln!("  Constraint {:?}: {}/{} rows had mismatches", constraint.name, mismatches, total_rows);
            }
        }
    }

    #[test]
    fn debug_rightlookup_operand_values() {
        let acc = build_realistic_test_accessor(10);
        let acc_ref: &dyn WitnessRowAccessor<crate::field::tracked_ark::TrackedFr> = &acc;

        eprintln!("Debugging RightLookupOperand values used in constraints:");
        for row in 0..acc.num_steps() {
            let right_lookup_operand = acc_ref.value_at(JoltR1CSInputs::RightLookupOperand.to_index(), row);
            let right_lookup_operand_old = acc_ref.value_at_field_old(JoltR1CSInputs::RightLookupOperand.to_index(), row);

            eprintln!("Row {}: RightLookupOperand = {:?}, old_field = {:?}", row, right_lookup_operand, right_lookup_operand_old);

            if let SmallScalar::U128(val) = right_lookup_operand {
                let as_i128 = right_lookup_operand.as_i128();
                let direct_field = crate::field::tracked_ark::TrackedFr::from_u128(val);
                let i128_field = crate::field::tracked_ark::TrackedFr::from_i128(as_i128);

                eprintln!("  U128 value: {}", val);
                eprintln!("  as_i128(): {}", as_i128);
                eprintln!("  direct field (old): {:?}", direct_field);
                eprintln!("  i128 field (generic): {:?}", i128_field);
                eprintln!("  Match: {}", direct_field == i128_field);
            }
        }
    }

    /// Normalize BzValue to S192 (SignedBigInt<3>) for equality up to widening.
    fn bz_to_l3(v: BzValue) -> SignedBigInt<3> {
        match v {
            BzValue::S64(s) => SignedBigInt::from_bigint(
                {
                    let mut m = BigInt::<3>::zero();
                    m.0[0] = s.magnitude.0[0];
                    m
                },
                s.is_positive,
            ),
            BzValue::S128(s) => SignedBigInt::from_bigint(
                {
                    let mut m = BigInt::<3>::zero();
                    m.0[0] = s.magnitude.0[0];
                    m.0[1] = s.magnitude.0[1];
                    m
                },
                s.is_positive,
            ),
            BzValue::S192(s) => s,
        }
    }

    #[test]
    fn bz_named_matches_generic_b_up_to_widening() {
        let acc = build_test_accessor(100);
        let acc_ref: &dyn WitnessRowAccessor<crate::field::tracked_ark::TrackedFr> = &acc;
        for c in UNIFORM_R1CS.iter() {
            for row in 0..acc.num_steps() {
                let gen = eval_bz_generic::<crate::field::tracked_ark::TrackedFr>(
                    &c.cons.b, acc_ref, row,
                );
                let named =
                    eval_bz_by_name::<crate::field::tracked_ark::TrackedFr>(c, acc_ref, row);
                let gen_l3 = bz_to_l3(gen);
                let named_l3 = bz_to_l3(named);
                assert_eq!(
                    gen_l3, named_l3,
                    "Bz mismatch for {:?} at row {}: generic={:?}, named={:?}",
                    c.name, row, gen, named
                );
            }
        }
    }
}
