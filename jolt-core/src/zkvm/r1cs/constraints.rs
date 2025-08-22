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
//! 3. Optionally (but encouraged) add a custom evaluator in `eval_az_bz_by_name`
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
//! Custom evaluators in `eval_az_bz_by_name` provide optimized Az/Bz evaluation
//! using `SmallScalar` types to avoid field conversions. They should:
//! - Use appropriate `AzValue` variants (prefer `I8` for flags/small sums)
//! - Use appropriate `BzValue` variants (prefer `U64AndSign` for u64-u64 diffs)
//! - Only use `U128AndSign` when 128-bit arithmetic is inherently required

use super::inputs::{JoltR1CSInputs, WitnessRowAccessor};
use crate::field::JoltField;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::zkvm::instruction::CircuitFlags;

// Re-export key types from ops module for convenience
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
        accessor: &dyn WitnessRowAccessor<F>,
        row: usize,
    ) -> F {
        let mut result = F::zero();
        self.for_each_term(|input_index, coeff| {
            let field_coeff = F::from_i128(coeff);
            result += accessor.value_at(input_index, row).mul_field(field_coeff);
        });
        if let Some(c) = self.const_term() {
            result += F::from_i128(c);
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
    CompressedNoUpdateDef,
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
pub const NUM_R1CS_CONSTRAINTS: usize = 28;

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
        ({ JoltR1CSInputs::RightInstructionInput }) * ({ JoltR1CSInputs::LeftInstructionInput })
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
    r1cs_prod!(
        name: ConstraintName::CompressedNoUpdateDef,
        ({ JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed) })
            * ({ JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC) })
            == ({ JoltR1CSInputs::CompressedDoNotUpdateUnexpPC })
    ),
    r1cs_eq_conditional!(
        name: ConstraintName::NextUnexpPCUpdateOtherwise,
        if { { 1i128 } - { JoltR1CSInputs::ShouldBranch } - { JoltR1CSInputs::OpFlags(CircuitFlags::Jump) } }
        => ( { JoltR1CSInputs::NextUnexpandedPC } )
           == ( { JoltR1CSInputs::UnexpandedPC } + { 4i128 }
                - { 4 * JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC) }
                - { 2 * JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed) }
                + { 2 * JoltR1CSInputs::CompressedDoNotUpdateUnexpPC } )
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
    let abs = (diff as i128).unsigned_abs();
    if abs as u128 <= u64::MAX as u128 {
        BzValue::S64(SignedBigInt::from_u64_with_sign(abs as u64, diff >= 0))
    } else {
        BzValue::S128(SignedBigInt::<2>::from_i128(diff))
    }
}

#[inline]
fn flag_to_az(flag: bool) -> AzValue {
    AzValue::I8(if flag { 1 } else { 0 })
}

pub fn eval_az_bz_by_name<F: JoltField>(
    c: &NamedConstraint,
    accessor: &dyn WitnessRowAccessor<F>,
    row: usize,
) -> Option<(AzValue, BzValue)> {
    use JoltR1CSInputs as Inp;
    match c.name {
        ConstraintName::LeftInputEqRs1 => {
            // Az: single flag (0 or 1) -> I8
            let az = flag_to_az(matches!(
                accessor.value_at(
                    Inp::OpFlags(CircuitFlags::LeftOperandIsRs1Value).to_index(),
                    row
                ),
                SmallScalar::Bool(true)
            ));
            // Bz: u64 - u64 difference -> U64AndSign (avoids i128 for simple subtraction)
            let left = accessor
                .value_at(Inp::LeftInstructionInput.to_index(), row)
                .as_u64_clamped();
            let rs1 = accessor
                .value_at(Inp::Rs1Value.to_index(), row)
                .as_u64_clamped();
            let bz = BzValue::S64(SignedBigInt::from_u64_with_sign(
                if left >= rs1 { left - rs1 } else { rs1 - left },
                left >= rs1,
            ));
            Some((az, bz))
        }
        ConstraintName::LeftInputEqPC => {
            // Az: single flag (0 or 1) -> I8
            let az = flag_to_az(matches!(
                accessor
                    .value_at(Inp::OpFlags(CircuitFlags::LeftOperandIsPC).to_index(), row),
                SmallScalar::Bool(true)
            ));
            // Bz: u64 - u64 difference -> U64AndSign
            let left = accessor
                .value_at(Inp::LeftInstructionInput.to_index(), row)
                .as_u64_clamped();
            let pc = accessor
                .value_at(Inp::UnexpandedPC.to_index(), row)
                .as_u64_clamped();
            let bz = BzValue::S64(SignedBigInt::from_u64_with_sign(
                if left >= pc { left - pc } else { pc - left },
                left >= pc,
            ));
            Some((az, bz))
        }
        ConstraintName::RightInputEqRs2 => {
            // Az: single flag (0 or 1) -> I8
            let az = flag_to_az(matches!(
                accessor.value_at(
                    Inp::OpFlags(CircuitFlags::RightOperandIsRs2Value).to_index(),
                    row
                ),
                SmallScalar::Bool(true)
            ));
            // Bz: i64 - u64 difference -> use i128 since RightInstructionInput can be negative
            let right = accessor
                .value_at(Inp::RightInstructionInput.to_index(), row)
                .as_i128();
            let rs2 = accessor
                .value_at(Inp::Rs2Value.to_index(), row)
                .as_i128();
            let bz = diff_to_bz(right - rs2);
            Some((az, bz))
        }
        ConstraintName::RightInputEqImm => {
            // Az: single flag (0 or 1) -> I8
            let az = flag_to_az(matches!(
                accessor.value_at(
                    Inp::OpFlags(CircuitFlags::RightOperandIsImm).to_index(),
                    row
                ),
                SmallScalar::Bool(true)
            ));
            // Bz: i64 - i128 difference -> use i128 since both can be negative
            let right = accessor
                .value_at(Inp::RightInstructionInput.to_index(), row)
                .as_i128();
            let imm = accessor.value_at(Inp::Imm.to_index(), row).as_i128();
            let bz = diff_to_bz(right - imm);
            Some((az, bz))
        }
        ConstraintName::RamReadEqRamWriteIfLoad => {
            // Az: single flag (0 or 1) -> I8
            let az = flag_to_az(matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::Load).to_index(), row),
                SmallScalar::Bool(true)
            ));
            // Bz: u64 - u64 difference -> U64AndSign (both RAM values are u64)
            let rd = accessor
                .value_at(Inp::RamReadValue.to_index(), row)
                .as_u64_clamped();
            let wr = accessor
                .value_at(Inp::RamWriteValue.to_index(), row)
                .as_u64_clamped();
            let bz = BzValue::S64(SignedBigInt::from_u64_with_sign(
                if rd >= wr { rd - wr } else { wr - rd },
                rd >= wr,
            ));
            Some((az, bz))
        }
        ConstraintName::RamReadEqRdWriteIfLoad => {
            // Az: single flag (0 or 1) -> I8
            let az = flag_to_az(matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::Load).to_index(), row),
                SmallScalar::Bool(true)
            ));
            // Bz: u64 - u64 difference -> U64AndSign (both register values are u64)
            let rd = accessor
                .value_at(Inp::RamReadValue.to_index(), row)
                .as_u64_clamped();
            let rdw = accessor
                .value_at(Inp::RdWriteValue.to_index(), row)
                .as_u64_clamped();
            let bz = BzValue::S64(SignedBigInt::from_u64_with_sign(
                if rd >= rdw { rd - rdw } else { rdw - rd },
                rd >= rdw,
            ));
            Some((az, bz))
        }
        ConstraintName::LeftInputZeroOtherwise => {
            // Az: 1 - flag1 - flag2 -> I8 (mutually exclusive flags, so result is 0 or 1)
            let f1 = matches!(
                accessor.value_at(
                    Inp::OpFlags(CircuitFlags::LeftOperandIsRs1Value).to_index(),
                    row
                ),
                SmallScalar::Bool(true)
            );
            let f2 = matches!(
                accessor
                    .value_at(Inp::OpFlags(CircuitFlags::LeftOperandIsPC).to_index(), row),
                SmallScalar::Bool(true)
            );
            let az = AzValue::I8(1 - (f1 as i8) - (f2 as i8));
            // Bz: LeftInstructionInput (u64) -> U64 (no subtraction, just the value)
            let left = accessor
                .value_at(Inp::LeftInstructionInput.to_index(), row)
                .as_u64_clamped();
            let bz = BzValue::U64(left);
            Some((az, bz))
        }
        ConstraintName::RamAddrEqRs1PlusImmIfLoadStore => {
            // Az: Load OR Store flag -> I8 (0 or 1)
            let load = matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::Load).to_index(), row),
                SmallScalar::Bool(true)
            );
            let store = matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::Store).to_index(), row),
                SmallScalar::Bool(true)
            );
            let az = flag_to_az(load || store);
            // Bz: Rs1Value + Imm -> use i128 since Imm can be negative
            let rs1 = accessor
                .value_at(Inp::Rs1Value.to_index(), row)
                .as_i128();
            let imm = accessor.value_at(Inp::Imm.to_index(), row).as_i128();
            let bz = diff_to_bz(rs1 + imm);
            Some((az, bz))
        }
        ConstraintName::LeftLookupZeroUnlessAddSubMul => {
            // Az: sum of flags (clamped to 0 or 1) -> I8
            let add = matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::AddOperands).to_index(), row),
                SmallScalar::Bool(true)
            );
            let sub = matches!(
                accessor
                    .value_at(Inp::OpFlags(CircuitFlags::SubtractOperands).to_index(), row),
                SmallScalar::Bool(true)
            );
            let mul = matches!(
                accessor
                    .value_at(Inp::OpFlags(CircuitFlags::MultiplyOperands).to_index(), row),
                SmallScalar::Bool(true)
            );
            let cond = ((add as i8) + (sub as i8) + (mul as i8)).clamp(0, 1);
            let az = AzValue::I8(cond);
            // Bz: 0 - LeftInstructionInput -> negate u64 value
            let left = accessor
                .value_at(Inp::LeftInstructionInput.to_index(), row)
                .as_i128();
            let bz = diff_to_bz(0 - left);
            Some((az, bz))
        }
        ConstraintName::RightLookupAdd => {
            // Az: single flag (0 or 1) -> I8
            let az = flag_to_az(matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::AddOperands).to_index(), row),
                SmallScalar::Bool(true)
            ));
            // Bz: u128 - (u64 + i64) difference -> use i128 for mixed arithmetic
            let rlookup = accessor
                .value_at(Inp::RightLookupOperand.to_index(), row)
                .as_i128();
            let left = accessor
                .value_at(Inp::LeftInstructionInput.to_index(), row)
                .as_i128();
            let right = accessor
                .value_at(Inp::RightInstructionInput.to_index(), row)
                .as_i128();
            let bz = diff_to_bz(rlookup - (left + right));
            Some((az, bz))
        }
        ConstraintName::RightLookupSub => {
            // Az: single flag (0 or 1) -> I8
            let az = flag_to_az(matches!(
                accessor
                    .value_at(Inp::OpFlags(CircuitFlags::SubtractOperands).to_index(), row),
                SmallScalar::Bool(true)
            ));
            // Bz: u128 - (u64 - i64 + 2^64) -> use i128 for twos-complement conversion
            let rlookup = accessor
                .value_at(Inp::RightLookupOperand.to_index(), row)
                .as_i128();
            let left = accessor
                .value_at(Inp::LeftInstructionInput.to_index(), row)
                .as_i128();
            let right = accessor
                .value_at(Inp::RightInstructionInput.to_index(), row)
                .as_i128();
            let two64 = (u64::MAX as u128 + 1) as i128;
            let target = left - right + two64;
            let bz = diff_to_bz(rlookup - target);
            Some((az, bz))
        }
        ConstraintName::ProductDef => {
            // Az: LeftInstructionInput (u64) -> U64AndSign (no sign conversion needed)
            let left = accessor
                .value_at(Inp::LeftInstructionInput.to_index(), row)
                .as_u64_clamped();
            let right_i128 = accessor
                .value_at(Inp::RightInstructionInput.to_index(), row)
                .as_i128();
            let az = AzValue::S64(SignedBigInt::from_u64_with_sign(left, true));
            // Bz: RightInstructionInput (i64) -> U64AndSign with explicit sign handling
            let bz = if right_i128 >= 0 {
                BzValue::S64(SignedBigInt::from_u64_with_sign(right_i128 as u64, true))
            } else {
                BzValue::S64(SignedBigInt::from_u64_with_sign((-right_i128) as u64, false))
            };
            Some((az, bz))
        }
        ConstraintName::RightLookupEqProductIfMul => {
            // Az: single flag (0 or 1) -> I8
            let az = flag_to_az(matches!(
                accessor
                    .value_at(Inp::OpFlags(CircuitFlags::MultiplyOperands).to_index(), row),
                SmallScalar::Bool(true)
            ));
            // Bz: u128 - u128 difference -> use i128 for 128-bit arithmetic
            let rlookup = accessor
                .value_at(Inp::RightLookupOperand.to_index(), row)
                .as_i128();
            let prod = accessor
                .value_at(Inp::Product.to_index(), row)
                .as_i128();
            let bz = diff_to_bz(rlookup - prod);
            Some((az, bz))
        }
        ConstraintName::RightLookupEqRightInputOtherwise => {
            // Az: 1 - sum of 4 flags -> I8 (negated sum of mutually exclusive flags)
            let one = 1i8;
            let add = matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::AddOperands).to_index(), row),
                SmallScalar::Bool(true)
            ) as i8;
            let sub = matches!(
                accessor
                    .value_at(Inp::OpFlags(CircuitFlags::SubtractOperands).to_index(), row),
                SmallScalar::Bool(true)
            ) as i8;
            let mul = matches!(
                accessor
                    .value_at(Inp::OpFlags(CircuitFlags::MultiplyOperands).to_index(), row),
                SmallScalar::Bool(true)
            ) as i8;
            let adv = matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::Advice).to_index(), row),
                SmallScalar::Bool(true)
            ) as i8;
            let az = AzValue::I8(one - add - sub - mul - adv);
            // Bz: u128 - i64 difference -> use i128 for mixed arithmetic
            let rlookup = accessor
                .value_at(Inp::RightLookupOperand.to_index(), row)
                .as_i128();
            let right = accessor
                .value_at(Inp::RightInstructionInput.to_index(), row)
                .as_i128();
            let bz = diff_to_bz(rlookup - right);
            Some((az, bz))
        }
        ConstraintName::AssertLookupOne => {
            // Az: single flag (0 or 1) -> I8
            let az = flag_to_az(matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::Assert).to_index(), row),
                SmallScalar::Bool(true)
            ));
            // Bz: u64 - 1 difference -> use i128 to handle potential negative result
            let lookup = accessor
                .value_at(Inp::LookupOutput.to_index(), row)
                .as_i128();
            let bz = diff_to_bz(lookup - 1);
            Some((az, bz))
        }
        ConstraintName::WriteLookupOutputToRDDef => {
            // Az: Rd register index (u8) -> I8 (register indices are small)
            let rd = accessor.value_at(Inp::Rd.to_index(), row);
            let az = match rd {
                SmallScalar::U8(v) => AzValue::I8(v as i8),
                _ => AzValue::I8(0),
            };
            // Bz: flag (0 or 1) -> U64 (simple flag value)
            let flag = matches!(
                accessor.value_at(
                    Inp::OpFlags(CircuitFlags::WriteLookupOutputToRD).to_index(),
                    row
                ),
                SmallScalar::Bool(true)
            );
            let bz = BzValue::U64(if flag { 1 } else { 0 });
            Some((az, bz))
        }
        ConstraintName::RdWriteEqLookupIfWriteLookupToRd => {
            // Az: single flag (0 or 1) -> I8
            let az = flag_to_az(matches!(
                accessor.value_at(Inp::WriteLookupOutputToRD.to_index(), row),
                SmallScalar::U8(1)
            ));
            // Bz: u64 - u64 difference -> U64AndSign (both register and lookup are u64)
            let rdw = accessor
                .value_at(Inp::RdWriteValue.to_index(), row)
                .as_u64_clamped();
            let lookup = accessor
                .value_at(Inp::LookupOutput.to_index(), row)
                .as_u64_clamped();
            let bz = BzValue::S64(SignedBigInt::from_u64_with_sign(
                if rdw >= lookup {
                    rdw - lookup
                } else {
                    lookup - rdw
                },
                rdw >= lookup,
            ));
            Some((az, bz))
        }
        ConstraintName::WritePCtoRDDef => {
            // Az: Rd register index (u8) -> I8 (register indices are small)
            let rd = accessor.value_at(Inp::Rd.to_index(), row);
            let az = match rd {
                SmallScalar::U8(v) => AzValue::I8(v as i8),
                _ => AzValue::I8(0),
            };
            // Bz: flag (0 or 1) -> U64 (simple flag value)
            let jump = matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::Jump).to_index(), row),
                SmallScalar::Bool(true)
            );
            let bz = BzValue::U64(if jump { 1 } else { 0 });
            Some((az, bz))
        }
        ConstraintName::RdWriteEqPCPlusConstIfWritePCtoRD => {
            let az = flag_to_az(matches!(
                accessor.value_at(Inp::WritePCtoRD.to_index(), row),
                SmallScalar::Bool(true)
            ));
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
            let bz = diff_to_bz(rdw - (pc + const_term as i128));
            Some((az, bz))
        }
        ConstraintName::ShouldJumpDef => {
            let jump = matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::Jump).to_index(), row),
                SmallScalar::Bool(true)
            );
            let az = flag_to_az(jump);
            let next_noop = matches!(
                accessor.value_at(Inp::NextIsNoop.to_index(), row),
                SmallScalar::Bool(true)
            );
            let bz = BzValue::U64(if next_noop { 0 } else { 1 });
            Some((az, bz))
        }
        ConstraintName::NextUnexpPCEqLookupIfShouldJump => {
            let az = flag_to_az(matches!(
                accessor.value_at(Inp::ShouldJump.to_index(), row),
                SmallScalar::U8(1)
            ));
            let nextpc = accessor
                .value_at(Inp::NextUnexpandedPC.to_index(), row)
                .as_i128();
            let lookup = accessor
                .value_at(Inp::LookupOutput.to_index(), row)
                .as_i128();
            let bz = diff_to_bz(nextpc - lookup);
            Some((az, bz))
        }
        ConstraintName::ShouldBranchDef => {
            let az = flag_to_az(matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::Branch).to_index(), row),
                SmallScalar::Bool(true)
            ));
            let lookup = accessor
                .value_at(Inp::LookupOutput.to_index(), row)
                .as_u64_clamped();
            let bz = BzValue::U64(lookup);
            Some((az, bz))
        }
        ConstraintName::NextUnexpPCEqPCPlusImmIfShouldBranch => {
            let az = flag_to_az(matches!(
                accessor.value_at(Inp::ShouldBranch.to_index(), row),
                SmallScalar::U8(1)
            ));
            let next = accessor
                .value_at(Inp::NextUnexpandedPC.to_index(), row)
                .as_i128();
            let pc = accessor
                .value_at(Inp::UnexpandedPC.to_index(), row)
                .as_i128();
            let imm = accessor.value_at(Inp::Imm.to_index(), row).as_i128();
            let bz = diff_to_bz(next - (pc + imm));
            Some((az, bz))
        }
        ConstraintName::CompressedNoUpdateDef => {
            let az = flag_to_az(matches!(
                accessor.value_at(Inp::OpFlags(CircuitFlags::IsCompressed).to_index(), row),
                SmallScalar::Bool(true)
            ));
            let flag2 = matches!(
                accessor.value_at(
                    Inp::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC).to_index(),
                    row
                ),
                SmallScalar::Bool(true)
            );
            let bz = BzValue::U64(if flag2 { 1 } else { 0 });
            Some((az, bz))
        }
        ConstraintName::NextUnexpPCUpdateOtherwise => {
            let cond_i8 = 1
                - (matches!(
                    accessor.value_at(Inp::ShouldBranch.to_index(), row),
                    SmallScalar::U8(1)
                ) as i8)
                - (matches!(
                    accessor.value_at(Inp::OpFlags(CircuitFlags::Jump).to_index(), row),
                    SmallScalar::Bool(true)
                ) as i8);
            let az = AzValue::I8(cond_i8);
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
            let comp_no_upd = matches!(
                accessor.value_at(Inp::CompressedDoNotUpdateUnexpPC.to_index(), row),
                SmallScalar::U8(1)
            ) as i128;
            let target = pc + 4 - 4 * dnoupd - 2 * iscompr + 2 * comp_no_upd;
            let bz = diff_to_bz(next - target);
            Some((az, bz))
        }
        ConstraintName::NextPCEqPCPlusOneIfInline => {
            let az = flag_to_az(matches!(
                accessor.value_at(
                    Inp::OpFlags(CircuitFlags::InlineSequenceInstruction).to_index(),
                    row
                ),
                SmallScalar::Bool(true)
            ));
            let next = accessor
                .value_at(Inp::NextPC.to_index(), row)
                .as_i128();
            let pc = accessor.value_at(Inp::PC.to_index(), row).as_i128();
            let bz = diff_to_bz(next - (pc + 1));
            Some((az, bz))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
                "ConstraintName variant {:?} not used in UNIFORM_R1CS",
                variant
            );
        }

        // Ensure enum variant count matches NUM_R1CS_CONSTRAINTS
        let variant_count = ConstraintName::iter().count();
        assert_eq!(
            variant_count, NUM_R1CS_CONSTRAINTS,
            "ConstraintName variant count ({}) doesn't match NUM_R1CS_CONSTRAINTS ({})",
            variant_count, NUM_R1CS_CONSTRAINTS
        );
    }
}
