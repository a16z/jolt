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
    RightLookupEqProductIfMul,
    RightLookupEqRightInputOtherwise,
    AssertLookupOne,
    WriteLookupOutputToRDDef,
    RdWriteEqLookupIfWriteLookupToRd,
    WritePCtoRDDef,
    RdWriteEqPCPlusConstIfWritePCtoRD,
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
pub const NUM_R1CS_CONSTRAINTS: usize = 25;

/// Static table of all 25 R1CS uniform constraints.
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

/// Evaluate Az by name using a fully materialized R1CS cycle inputs
pub fn eval_az_by_name<F: JoltField>(c: &NamedConstraint, row: &R1CSCycleInputs) -> I8OrI96 {
    use ConstraintName as N;
    match c.name {
        // Az: LeftOperandIsRs1Value flag (0/1)
        N::LeftInputEqRs1 => row.flags[CircuitFlags::LeftOperandIsRs1Value].into(),
        // Az: LeftOperandIsPC flag (0/1)
        N::LeftInputEqPC => row.flags[CircuitFlags::LeftOperandIsPC].into(),
        N::LeftInputZeroOtherwise => {
            // NOTE: relies on exclusivity of these circuit flags (validated in tests):
            // return 1 only if neither flag is set
            let f1 = row.flags[CircuitFlags::LeftOperandIsRs1Value];
            let f2 = row.flags[CircuitFlags::LeftOperandIsPC];
            (!(f1 || f2)).into()
        }
        // Az: RightOperandIsRs2Value flag (0/1)
        N::RightInputEqRs2 => row.flags[CircuitFlags::RightOperandIsRs2Value].into(),
        // Az: RightOperandIsImm flag (0/1)
        N::RightInputEqImm => row.flags[CircuitFlags::RightOperandIsImm].into(),
        N::RightInputZeroOtherwise => {
            // NOTE: relies on exclusivity of these circuit flags (validated in tests):
            // return 1 only if neither flag is set
            let f1 = row.flags[CircuitFlags::RightOperandIsRs2Value];
            let f2 = row.flags[CircuitFlags::RightOperandIsImm];
            (!(f1 || f2)).into()
        }
        N::RamAddrEqRs1PlusImmIfLoadStore => {
            // Az: Load OR Store flag (0/1)
            (row.flags[CircuitFlags::Load] || row.flags[CircuitFlags::Store]).into()
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
        // Az: Rd register index (0 disables write)
        N::WriteLookupOutputToRDDef => I8OrI96::from_i8(row.rd_addr as i8),
        N::RdWriteEqLookupIfWriteLookupToRd => {
            // Az: WriteLookupOutputToRD indicator (0/1)
            I8OrI96::from_i8(row.write_lookup_output_to_rd_addr as i8)
        }
        // Az: Rd register index (0 disables write)
        N::WritePCtoRDDef => I8OrI96::from_i8(row.rd_addr as i8),
        // Az: WritePCtoRD indicator (0/1)
        N::RdWriteEqPCPlusConstIfWritePCtoRD => I8OrI96::from_i8(row.write_pc_to_rd_addr as i8),
        // Az: ShouldJump indicator (0/1)
        N::NextUnexpPCEqLookupIfShouldJump => row.should_jump.into(),
        // Az: Branch flag (0/1)
        N::ShouldBranchDef => row.flags[CircuitFlags::Branch].into(),
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
pub fn eval_bz_by_name<F: JoltField>(c: &NamedConstraint, row: &R1CSCycleInputs) -> S160 {
    use ConstraintName as N;
    match c.name {
        // B: LeftInstructionInput - Rs1Value (signed-magnitude over u64 bit patterns)
        N::LeftInputEqRs1 => S160::from_diff_u64(row.left_input, row.rs1_read_value),
        // B: LeftInstructionInput - UnexpandedPC (signed-magnitude over u64 bit patterns)
        N::LeftInputEqPC => S160::from_diff_u64(row.left_input, row.unexpanded_pc),
        // B: LeftInstructionInput - 0 (u64 bit pattern)
        N::LeftInputZeroOtherwise => S160::from_diff_u64(row.left_input, 0),
        // B: RightInstructionInput - Rs2Value (i128 arithmetic)
        N::RightInputEqRs2 => S160::from(row.right_input) - S160::from(row.rs2_read_value),
        // B: RightInstructionInput - Imm (i128 arithmetic)
        N::RightInputEqImm => S160::from(row.right_input) - S160::from(row.imm),
        // B: RightInstructionInput - 0 (i128 arithmetic)
        N::RightInputZeroOtherwise => S160::from(row.right_input),
        N::RamAddrEqRs1PlusImmIfLoadStore => {
            // B: (Rs1Value + Imm) - 0 (true_val - false_val from if-else)
            if row.imm.is_positive {
                S160::from(row.rs1_read_value as u128 + row.imm.magnitude_as_u64() as u128)
            } else {
                S160::from(row.rs1_read_value as i128 - row.imm.magnitude_as_u64() as i128)
            }
        }
        // B: RamReadValue - RamWriteValue (u64 bit-pattern difference)
        N::RamReadEqRamWriteIfLoad => S160::from_diff_u64(row.ram_read_value, row.ram_write_value),
        // B: RamReadValue - RdWriteValue (u64 bit-pattern difference)
        N::RamReadEqRdWriteIfLoad => S160::from_diff_u64(row.ram_read_value, row.rd_write_value),
        // B: Rs2Value - RamWriteValue (u64 bit-pattern difference)
        N::Rs2EqRamWriteIfStore => S160::from_diff_u64(row.rs2_read_value, row.ram_write_value),
        // B: 0 - LeftInstructionInput (true_val - false_val from if-else)
        N::LeftLookupZeroUnlessAddSubMul => -S160::from(row.left_input),
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
        N::WriteLookupOutputToRDDef => {
            // B: OpFlags(WriteLookupOutputToRD) (boolean 0/1)
            if row.flags[CircuitFlags::WriteLookupOutputToRD] {
                S160::one()
            } else {
                S160::zero()
            }
        }
        N::RdWriteEqLookupIfWriteLookupToRd => {
            // B: RdWriteValue - LookupOutput (u64 bit-pattern difference)
            S160::from_diff_u64(row.rd_write_value, row.lookup_output)
        }
        N::WritePCtoRDDef => {
            // B: OpFlags(Jump) (boolean 0/1)
            if row.flags[CircuitFlags::Jump] {
                S160::one()
            } else {
                S160::zero()
            }
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
            // Note: B uses u64 bit-pattern difference here (matches accessor variant)
            // B: NextUnexpandedPC - LookupOutput (i128 arithmetic)
            S160::from_diff_u64(row.next_unexpanded_pc, row.lookup_output)
        }
        // B: LookupOutput (u64 bit pattern)
        N::ShouldBranchDef => S160::from(row.lookup_output),
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

// =============================================================================
// Batch evaluation functions
// =============================================================================

/// Batched evaluation using a fully materialized R1CS cycle inputs. This avoids any repeated
/// reads from the trace or bytecode and computes all constraints.
pub fn eval_az_bz_batch_from_row<F: JoltField>(
    constraints: &[NamedConstraint],
    row: &R1CSCycleInputs,
    az_output: &mut [I8OrI96],
    bz_output: &mut [S160],
) {
    assert_eq!(constraints.len(), az_output.len());
    assert_eq!(constraints.len(), bz_output.len());
    for (i, constraint) in constraints.iter().enumerate() {
        az_output[i] = eval_az_by_name::<F>(constraint, row);
        bz_output[i] = eval_bz_by_name::<F>(constraint, row);
    }
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
}
