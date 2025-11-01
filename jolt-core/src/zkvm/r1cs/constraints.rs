//! Uniform R1CS constraints and product virtualization
//!
//! This module contains only compile-time data and helpers that describe the
//! zkVM's uniform R1CS constraints and how they are grouped for the
//! univariate‑skip protocol, plus the constraints used by the product virtualization sumcheck.
//!
//! What lives here (compile-time only):
//! - The uniform equality-conditional constraints:
//!   - `R1CSConstraint`, `NamedR1CSConstraint`, and the `r1cs_eq_conditional!` macro
//!   - The canonical table `R1CS_CONSTRAINTS` and its name enum `R1CSConstraintName`
//! - The univariate‑skip split of the uniform constraints into two groups:
//!   - Constants `UNIVARIATE_SKIP_*` describing degree/domain sizes
//!   - The first-group selector `R1CS_CONSTRAINTS_FIRST_GROUP_NAMES` and its
//!     order-preserving filtered view `R1CS_CONSTRAINTS_FIRST_GROUP`
//!   - The compile-time complement `R1CS_CONSTRAINTS_SECOND_GROUP_NAMES` and view
//!     `R1CS_CONSTRAINTS_SECOND_GROUP`
//!   - Invariants: the first group has size `UNIVARIATE_SKIP_DOMAIN_SIZE` and is
//!     never smaller than the second; order matches `R1CS_CONSTRAINTS`
//! - The product virtualization constraints:
//!   - `ProductFactorExpr`, `ProductConstraint`, and the ordered list
//!     `PRODUCT_CONSTRAINTS`
//!   - Each row describes a factorization `Az · Bz = z'`, where `z'` names a
//!     `VirtualPolynomial` opening provided by Spartan outer; the order matches the
//!     product virtualization stage
//!
//! What does not live here:
//! - Runtime evaluation of any constraint (see `r1cs::evaluation` for typed
//!   Az/Bz evaluators, folding helpers, and product-virtualization evaluators)
//!
//! Notes for maintainers:
//! - When adding/removing a uniform constraint, keep `R1CSConstraintName`,
//!   `R1CS_CONSTRAINTS`, and the first-group name list in sync. The second group is
//!   computed as a complement at compile time.
//! - The first group is optimized for boolean guards and ~64-bit magnitudes;
//!   the second group is the remainder (e.g., wider Bz arithmetic).
//! - If you change guard/magnitude shapes, also update the typed evaluators in
//!   `r1cs::evaluation`.

use super::inputs::JoltR1CSInputs;
use crate::zkvm::instruction::{CircuitFlags, InstructionFlags};
use crate::zkvm::witness::VirtualPolynomial;
use strum::EnumCount;
use strum_macros::{EnumCount, EnumIter};

pub use super::ops::{Term, LC};

/// A single R1CS constraint row
#[derive(Clone, Copy, Debug)]
pub struct R1CSConstraint {
    pub a: LC,
    pub b: LC,
    // No c needed for now, all eq-conditional constraints
    // pub c: LC,
}

impl R1CSConstraint {
    pub const fn new(a: LC, b: LC) -> Self {
        Self { a, b }
    }
}

/// Creates: condition * (left - right) == 0
pub const fn constraint_eq_conditional_lc(condition: LC, left: LC, right: LC) -> R1CSConstraint {
    R1CSConstraint::new(
        condition,
        match left.checked_sub(right) {
            Some(b) => b,
            None => LC::zero(),
        },
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, EnumCount, EnumIter)]
pub enum R1CSConstraintName {
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
pub struct NamedR1CSConstraint {
    pub name: R1CSConstraintName,
    pub cons: R1CSConstraint,
}

/// r1cs_eq_conditional!: verbose, condition-first equality constraint
///
/// Usage: `r1cs_eq_conditional!(name: R1CSConstraintName::Foo, if { COND } => { LEFT } == { RIGHT });`
#[macro_export]
macro_rules! r1cs_eq_conditional {
    (name: $nm:expr, if { $($cond:tt)* } => ( $($left:tt)* ) == ( $($right:tt)* ) ) => {{
        $crate::zkvm::r1cs::constraints::NamedR1CSConstraint {
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
pub const NUM_R1CS_CONSTRAINTS: usize = R1CSConstraintName::COUNT;

/// Static table of all R1CS uniform constraints, to be proven in Spartan outer sumcheck
pub static R1CS_CONSTRAINTS: [NamedR1CSConstraint; NUM_R1CS_CONSTRAINTS] = [
    // if Load || Store {
    //     assert!(RamAddress == Rs1Value + Imm)
    // } else {
    //     assert!(RamAddress == 0)
    // }
    r1cs_eq_conditional!(
        name: R1CSConstraintName::RamAddrEqRs1PlusImmIfLoadStore,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::Load) } + { JoltR1CSInputs::OpFlags(CircuitFlags::Store) } }
        => ( { JoltR1CSInputs::RamAddress } ) == ( { JoltR1CSInputs::Rs1Value } + { JoltR1CSInputs::Imm } )
    ),
    r1cs_eq_conditional!(
        name: R1CSConstraintName::RamAddrEqZeroIfNotLoadStore,
        if { { 1i128 } - { JoltR1CSInputs::OpFlags(CircuitFlags::Load) } - { JoltR1CSInputs::OpFlags(CircuitFlags::Store) } }
        => ( { JoltR1CSInputs::RamAddress } ) == ( { 0i128 } )
    ),
    // if Load {
    //     assert!(RamReadValue == RamWriteValue)
    // }
    r1cs_eq_conditional!(
        name: R1CSConstraintName::RamReadEqRamWriteIfLoad,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::Load) } }
        => ( { JoltR1CSInputs::RamReadValue } ) == ( { JoltR1CSInputs::RamWriteValue } )
    ),
    // if Load {
    //     assert!(RamReadValue == RdWriteValue)
    // }
    r1cs_eq_conditional!(
        name: R1CSConstraintName::RamReadEqRdWriteIfLoad,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::Load) } }
        => ( { JoltR1CSInputs::RamReadValue } ) == ( { JoltR1CSInputs::RdWriteValue } )
    ),
    // if Store {
    //     assert!(Rs2Value == RamWriteValue)
    // }
    r1cs_eq_conditional!(
        name: R1CSConstraintName::Rs2EqRamWriteIfStore,
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
        name: R1CSConstraintName::LeftLookupZeroUnlessAddSubMul,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands) } + { JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands) } + { JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands) } }
        => ( { JoltR1CSInputs::LeftLookupOperand } ) == ( { 0i128 } )
    ),
    r1cs_eq_conditional!(
        name: R1CSConstraintName::LeftLookupEqLeftInputOtherwise,
        if { { 1i128 } - { JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands) } - { JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands) } - { JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands) } }
        => ( { JoltR1CSInputs::LeftLookupOperand } ) == ( { JoltR1CSInputs::LeftInstructionInput } )
    ),
    // If AddOperands {
    //     assert!(RightLookupOperand == LeftInstructionInput + RightInstructionInput)
    // }
    r1cs_eq_conditional!(
        name: R1CSConstraintName::RightLookupAdd,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands) } }
        => ( { JoltR1CSInputs::RightLookupOperand } ) == ( { JoltR1CSInputs::LeftInstructionInput } + { JoltR1CSInputs::RightInstructionInput } )
    ),
    // If SubtractOperands {
    //     assert!(RightLookupOperand == LeftInstructionInput - RightInstructionInput)
    // }
    // Converts from unsigned to twos-complement representation
    r1cs_eq_conditional!(
        name: R1CSConstraintName::RightLookupSub,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands) } }
        => ( { JoltR1CSInputs::RightLookupOperand } ) == ( { JoltR1CSInputs::LeftInstructionInput } - { JoltR1CSInputs::RightInstructionInput } + { 0x10000000000000000i128 } )
    ),
    r1cs_eq_conditional!(
        name: R1CSConstraintName::RightLookupEqProductIfMul,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands) } }
        => ( { JoltR1CSInputs::RightLookupOperand } ) == ( { JoltR1CSInputs::Product } )
    ),
    // if !(AddOperands || SubtractOperands || MultiplyOperands || Advice) {
    //     assert!(RightLookupOperand == RightInstructionInput)
    // }
    // Arbitrary untrusted advice goes in right lookup operand
    r1cs_eq_conditional!(
        name: R1CSConstraintName::RightLookupEqRightInputOtherwise,
        if { { 1i128 } - { JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands) } - { JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands) } - { JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands) } - { JoltR1CSInputs::OpFlags(CircuitFlags::Advice) } }
        => ( { JoltR1CSInputs::RightLookupOperand } ) == ( { JoltR1CSInputs::RightInstructionInput } )
    ),
    // if Assert {
    //     assert!(LookupOutput == 1)
    // }
    r1cs_eq_conditional!(
        name: R1CSConstraintName::AssertLookupOne,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::Assert) } }
        => ( { JoltR1CSInputs::LookupOutput } ) == ( { 1i128 } )
    ),
    // if Rd != 0 && WriteLookupOutputToRD {
    //     assert!(RdWriteValue == LookupOutput)
    // }
    r1cs_eq_conditional!(
        name: R1CSConstraintName::RdWriteEqLookupIfWriteLookupToRd,
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
        name: R1CSConstraintName::RdWriteEqPCPlusConstIfWritePCtoRD,
        if { { JoltR1CSInputs::WritePCtoRD } }
        => ( { JoltR1CSInputs::RdWriteValue } ) == ( { JoltR1CSInputs::UnexpandedPC } + { 4i128 } - { 2 * JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed) } )
    ),
    // if Jump && !NextIsNoop {
    //     assert!(NextUnexpandedPC == LookupOutput)
    // }
    r1cs_eq_conditional!(
        name: R1CSConstraintName::NextUnexpPCEqLookupIfShouldJump,
        if { { JoltR1CSInputs::ShouldJump } }
        => ( { JoltR1CSInputs::NextUnexpandedPC } ) == ( { JoltR1CSInputs::LookupOutput } )
    ),
    // if Branch && LookupOutput {
    //     assert!(NextUnexpandedPC == UnexpandedPC + Imm)
    // }
    r1cs_eq_conditional!(
        name: R1CSConstraintName::NextUnexpPCEqPCPlusImmIfShouldBranch,
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
        name: R1CSConstraintName::NextUnexpPCUpdateOtherwise,
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
        name: R1CSConstraintName::NextPCEqPCPlusOneIfInline,
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::VirtualInstruction) } }
        => ( { JoltR1CSInputs::NextPC } ) == ( { JoltR1CSInputs::PC } + { 1i128 } )
    ),
    // if NextIsVirtual && !NextIsFirstInSequence {
    //     assert!(1 == DoNotUpdateUnexpandedPC)
    // }
    // (note: we put 1 on LHS to keep Bz boolean)
    r1cs_eq_conditional!(
        name: R1CSConstraintName::MustStartSequenceFromBeginning,
        if { { JoltR1CSInputs::NextIsVirtual } - { JoltR1CSInputs::NextIsFirstInSequence } }
        => ( { 1i128 } ) == ( { JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC) } )
    ),
];

// =============================================================================
// Univariate skip constants for Spartan outer sumcheck and grouped views
// =============================================================================

/// Degree of univariate skip, defined to be `(NUM_R1CS_CONSTRAINTS - 1) / 2`
pub const OUTER_UNIVARIATE_SKIP_DEGREE: usize = (NUM_R1CS_CONSTRAINTS - 1) / 2;

/// Domain size of univariate skip, defined to be `OUTER_UNIVARIATE_SKIP_DEGREE + 1`.
pub const OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE: usize = OUTER_UNIVARIATE_SKIP_DEGREE + 1;

/// Extended domain size of univariate skip, defined to be `2 * OUTER_UNIVARIATE_SKIP_DEGREE + 1`.
pub const OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE: usize = 2 * OUTER_UNIVARIATE_SKIP_DEGREE + 1;

/// Number of coefficients in the first-round polynomial, defined to be `3 * OUTER_UNIVARIATE_SKIP_DEGREE + 1`.
pub const OUTER_FIRST_ROUND_POLY_NUM_COEFFS: usize = 3 * OUTER_UNIVARIATE_SKIP_DEGREE + 1;

/// Degree of the first-round polynomial.
pub const OUTER_FIRST_ROUND_POLY_DEGREE_BOUND: usize = OUTER_FIRST_ROUND_POLY_NUM_COEFFS - 1;

/// Number of remaining R1CS constraints in the second group, defined to be
/// `NUM_R1CS_CONSTRAINTS - OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE`.
pub const NUM_REMAINING_R1CS_CONSTRAINTS: usize =
    NUM_R1CS_CONSTRAINTS - OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE;

/// Order-preserving, compile-time filter over `R1CS_CONSTRAINTS` by constraint names.
const fn contains_name<const N: usize>(
    names: &[R1CSConstraintName; N],
    name: R1CSConstraintName,
) -> bool {
    let mut i = 0;
    while i < N {
        if names[i] as u32 == name as u32 {
            return true;
        }
        i += 1;
    }
    false
}

/// Select constraints from `R1CS_CONSTRAINTS` whose names appear in `names`, preserving order.
pub const fn filter_R1CS_CONSTRAINTS<const N: usize>(
    names: &[R1CSConstraintName; N],
) -> [NamedR1CSConstraint; N] {
    let dummy = NamedR1CSConstraint {
        name: R1CSConstraintName::RamReadEqRamWriteIfLoad,
        cons: R1CSConstraint::new(LC::zero(), LC::zero()),
    };
    let mut out: [NamedR1CSConstraint; N] = [dummy; N];

    let mut o = 0;
    let mut i = 0;
    while i < NUM_R1CS_CONSTRAINTS {
        let cand = R1CS_CONSTRAINTS[i];
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
        panic!(
            "filter_R1CS_CONSTRAINTS: not all requested constraints were found in R1CS_CONSTRAINTS"
        );
    }
    out
}

/// Compute the complement of `R1CS_CONSTRAINTS_FIRST_GROUP_NAMES` within `R1CS_CONSTRAINTS`.
const fn complement_first_group_names() -> [R1CSConstraintName; NUM_REMAINING_R1CS_CONSTRAINTS] {
    let mut out: [R1CSConstraintName; NUM_REMAINING_R1CS_CONSTRAINTS] =
        [R1CSConstraintName::RamReadEqRamWriteIfLoad; NUM_REMAINING_R1CS_CONSTRAINTS];
    let mut o = 0;
    let mut i = 0;
    while i < NUM_R1CS_CONSTRAINTS {
        let cand = R1CS_CONSTRAINTS[i].name;
        if !contains_name(&R1CS_CONSTRAINTS_FIRST_GROUP_NAMES, cand) {
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
pub const R1CS_CONSTRAINTS_FIRST_GROUP_NAMES: [R1CSConstraintName;
    OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE] = [
    R1CSConstraintName::RamAddrEqZeroIfNotLoadStore,
    R1CSConstraintName::RamReadEqRamWriteIfLoad,
    R1CSConstraintName::RamReadEqRdWriteIfLoad,
    R1CSConstraintName::Rs2EqRamWriteIfStore,
    R1CSConstraintName::LeftLookupZeroUnlessAddSubMul,
    R1CSConstraintName::LeftLookupEqLeftInputOtherwise,
    R1CSConstraintName::AssertLookupOne,
    R1CSConstraintName::NextUnexpPCEqLookupIfShouldJump,
    R1CSConstraintName::NextPCEqPCPlusOneIfInline,
    R1CSConstraintName::MustStartSequenceFromBeginning,
];

/// Second group: complement of first within R1CS_CONSTRAINTS
/// Here, Az may be u8, and Bz may be around 128 bits
pub const R1CS_CONSTRAINTS_SECOND_GROUP_NAMES: [R1CSConstraintName;
    NUM_REMAINING_R1CS_CONSTRAINTS] = complement_first_group_names();

/// First group: 10 boolean-guarded eq constraints, where Bz is around 64 bits
pub static R1CS_CONSTRAINTS_FIRST_GROUP: [NamedR1CSConstraint; OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE] =
    filter_R1CS_CONSTRAINTS(&R1CS_CONSTRAINTS_FIRST_GROUP_NAMES);

/// Second group: complement of first within R1CS_CONSTRAINTS, where Az may be u8 and Bz may be around 128 bits
pub static R1CS_CONSTRAINTS_SECOND_GROUP: [NamedR1CSConstraint; NUM_REMAINING_R1CS_CONSTRAINTS] =
    filter_R1CS_CONSTRAINTS(&R1CS_CONSTRAINTS_SECOND_GROUP_NAMES);

// ===========================================
// Product virtualization constraints
// ===========================================

/// Domain sizing for product-virtualization univariate-skip (size-5 window)
pub const NUM_PRODUCT_VIRTUAL: usize = 5;
pub const PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE: usize = NUM_PRODUCT_VIRTUAL;
pub const PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE: usize = NUM_PRODUCT_VIRTUAL - 1;
pub const PRODUCT_VIRTUAL_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE: usize =
    2 * PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE + 1;
pub const PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS: usize =
    3 * PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE + 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, EnumCount, EnumIter)]
pub enum ProductConstraintName {
    Instruction,
    WriteLookupOutputToRD,
    WritePCtoRD,
    ShouldBranch,
    ShouldJump,
}

/// Number of product virtualization constraints
pub const NUM_PRODUCT_CONSTRAINTS: usize = ProductConstraintName::COUNT;

/// Factor expression for product constraints: either a direct virtual polynomial,
/// or 1 minus a virtual polynomial (used for NextIsNoop in ShouldJump).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProductFactorExpr {
    Var(VirtualPolynomial),
    OneMinus(VirtualPolynomial),
}

/// A single product constraint row: Az · Bz = z', where z' is a virtual
/// polynomial whose claims come from Spartan outer's first stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProductConstraint {
    pub name: ProductConstraintName,
    pub left: ProductFactorExpr,
    pub right: ProductFactorExpr,
    pub output: VirtualPolynomial,
}

/// Canonical list of the product constraints in the same order as
/// `PRODUCT_VIRTUAL_TERMS` used by the product virtualization stage.
pub const PRODUCT_CONSTRAINTS: [ProductConstraint; NUM_PRODUCT_CONSTRAINTS] = [
    // 0: Product = LeftInstructionInput · RightInstructionInput
    ProductConstraint {
        name: ProductConstraintName::Instruction,
        left: ProductFactorExpr::Var(VirtualPolynomial::LeftInstructionInput),
        right: ProductFactorExpr::Var(VirtualPolynomial::RightInstructionInput),
        output: VirtualPolynomial::Product,
    },
    // 1: WriteLookupOutputToRD = IsRdNotZero · OpFlags(WriteLookupOutputToRD)
    ProductConstraint {
        name: ProductConstraintName::WriteLookupOutputToRD,
        left: ProductFactorExpr::Var(VirtualPolynomial::InstructionFlags(
            InstructionFlags::IsRdNotZero,
        )),
        right: ProductFactorExpr::Var(VirtualPolynomial::OpFlags(
            CircuitFlags::WriteLookupOutputToRD,
        )),
        output: VirtualPolynomial::WriteLookupOutputToRD,
    },
    // 2: WritePCtoRD = IsRdNotZero · OpFlags(Jump)
    ProductConstraint {
        name: ProductConstraintName::WritePCtoRD,
        left: ProductFactorExpr::Var(VirtualPolynomial::InstructionFlags(
            InstructionFlags::IsRdNotZero,
        )),
        right: ProductFactorExpr::Var(VirtualPolynomial::OpFlags(CircuitFlags::Jump)),
        output: VirtualPolynomial::WritePCtoRD,
    },
    // 3: ShouldBranch = LookupOutput · InstructionFlags(Branch)
    ProductConstraint {
        name: ProductConstraintName::ShouldBranch,
        left: ProductFactorExpr::Var(VirtualPolynomial::LookupOutput),
        right: ProductFactorExpr::Var(VirtualPolynomial::InstructionFlags(
            InstructionFlags::Branch,
        )),
        output: VirtualPolynomial::ShouldBranch,
    },
    // 4: ShouldJump = OpFlags(Jump) · (1 − NextIsNoop)
    ProductConstraint {
        name: ProductConstraintName::ShouldJump,
        left: ProductFactorExpr::Var(VirtualPolynomial::OpFlags(CircuitFlags::Jump)),
        right: ProductFactorExpr::OneMinus(VirtualPolynomial::NextIsNoop),
        output: VirtualPolynomial::ShouldJump,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use strum::IntoEnumIterator;

    /// Test that the constraint name enum matches the uniform R1CS order.
    #[test]
    fn constraint_enum_matches_R1CS_CONSTRAINTS_order() {
        let enum_order: Vec<R1CSConstraintName> = R1CSConstraintName::iter().collect();
        let array_order: Vec<R1CSConstraintName> =
            R1CS_CONSTRAINTS.iter().map(|nc| nc.name).collect();
        assert_eq!(array_order.len(), NUM_R1CS_CONSTRAINTS);
        assert_eq!(enum_order, array_order);
    }
}
