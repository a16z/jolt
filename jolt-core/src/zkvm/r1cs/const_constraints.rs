//! Compile-time constant R1CS constraints
//!
//! This module provides a static, compile-time representation of R1CS constraints
//! to replace the dynamic constraint building in the prover's hot path.

use super::inputs::JoltR1CSInputs;
use crate::zkvm::instruction::CircuitFlags;

/// Helper for JoltR1CSInputs to get indices
impl JoltR1CSInputs {
    /// Convert this input to a usable index
    pub const fn idx(self) -> usize {
        self.to_index_const()
    }
}

/// A single term in a linear combination: (input_index, coefficient)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConstTerm {
    pub input_index: usize,
    pub coeff: i128,
}

impl ConstTerm {
    pub const fn new(input_index: usize, coeff: i128) -> Self {
        Self { input_index, coeff }
    }
}

/// Const-friendly linear combination enum that can hold 0-5 terms
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstLC {
    Zero,
    Const(i128),
    Terms1([ConstTerm; 1]),
    Terms2([ConstTerm; 2]),
    Terms3([ConstTerm; 3]),
    Terms4([ConstTerm; 4]),
    Terms5([ConstTerm; 5]),
    Terms1Const([ConstTerm; 1], i128),
    Terms2Const([ConstTerm; 2], i128),
    Terms3Const([ConstTerm; 3], i128),
    Terms4Const([ConstTerm; 4], i128),
    Terms5Const([ConstTerm; 5], i128),
}

impl ConstLC {
    pub const fn zero() -> Self {
        ConstLC::Zero
    }

    pub const fn constant(value: i128) -> Self {
        ConstLC::Const(value)
    }

    pub const fn single_term(input_index: usize, coeff: i128) -> Self {
        ConstLC::Terms1([ConstTerm::new(input_index, coeff)])
    }

    pub fn num_terms(&self) -> usize {
        match self {
            ConstLC::Zero | ConstLC::Const(_) => 0,
            ConstLC::Terms1(_) | ConstLC::Terms1Const(_, _) => 1,
            ConstLC::Terms2(_) | ConstLC::Terms2Const(_, _) => 2,
            ConstLC::Terms3(_) | ConstLC::Terms3Const(_, _) => 3,
            ConstLC::Terms4(_) | ConstLC::Terms4Const(_, _) => 4,
            ConstLC::Terms5(_) | ConstLC::Terms5Const(_, _) => 5,
        }
    }

    pub fn term(&self, i: usize) -> Option<ConstTerm> {
        match self {
            ConstLC::Zero | ConstLC::Const(_) => None,
            ConstLC::Terms1(terms) | ConstLC::Terms1Const(terms, _) => terms.get(i).copied(),
            ConstLC::Terms2(terms) | ConstLC::Terms2Const(terms, _) => terms.get(i).copied(),
            ConstLC::Terms3(terms) | ConstLC::Terms3Const(terms, _) => terms.get(i).copied(),
            ConstLC::Terms4(terms) | ConstLC::Terms4Const(terms, _) => terms.get(i).copied(),
            ConstLC::Terms5(terms) | ConstLC::Terms5Const(terms, _) => terms.get(i).copied(),
        }
    }

    pub fn const_term(&self) -> Option<i128> {
        match self {
            ConstLC::Zero => None,
            ConstLC::Const(c) => Some(*c),
            ConstLC::Terms1(_)
            | ConstLC::Terms2(_)
            | ConstLC::Terms3(_)
            | ConstLC::Terms4(_)
            | ConstLC::Terms5(_) => None,
            ConstLC::Terms1Const(_, c)
            | ConstLC::Terms2Const(_, c)
            | ConstLC::Terms3Const(_, c)
            | ConstLC::Terms4Const(_, c)
            | ConstLC::Terms5Const(_, c) => Some(*c),
        }
    }

    /// Combine this ConstLC with another by addition
    /// Returns None if the result would exceed our term capacity (5 variable terms)
    pub const fn add_const_lc(self, other: ConstLC) -> Option<ConstLC> {
        let (mut out_terms, mut out_len, mut out_const) = Self::decompose(self);
        let (rhs_terms, rhs_len, rhs_const) = Self::decompose(other);

        out_const += rhs_const;

        let mut i = 0usize;
        while i < rhs_len {
            let term = rhs_terms[i];

            let mut found = false;
            let mut j = 0usize;
            while j < out_len {
                if out_terms[j].input_index == term.input_index {
                    let new_coeff = out_terms[j].coeff + term.coeff;
                    if new_coeff == 0 {
                        if out_len > 0 {
                            out_len -= 1;
                            out_terms[j] = out_terms[out_len];
                        }
                    } else {
                        out_terms[j] = ConstTerm::new(term.input_index, new_coeff);
                    }
                    found = true;
                    break;
                }
                j += 1;
            }

            if !found {
                if out_len >= 5 {
                    return None;
                }
                out_terms[out_len] = term;
                out_len += 1;
            }
            i += 1;
        }

        Some(Self::compose(&out_terms, out_len, out_const))
    }

    /// Break a ConstLC into (terms, len, const)
    const fn decompose(lc: ConstLC) -> ([ConstTerm; 5], usize, i128) {
        let mut terms = [ConstTerm {
            input_index: 0,
            coeff: 0,
        }; 5];
        let mut len = 0usize;
        let mut c = 0i128;
        match lc {
            ConstLC::Zero => {}
            ConstLC::Const(k) => {
                c = k;
            }
            ConstLC::Terms1([t1]) => {
                terms[0] = t1;
                len = 1;
            }
            ConstLC::Terms2([t1, t2]) => {
                terms[0] = t1;
                terms[1] = t2;
                len = 2;
            }
            ConstLC::Terms3([t1, t2, t3]) => {
                terms[0] = t1;
                terms[1] = t2;
                terms[2] = t3;
                len = 3;
            }
            ConstLC::Terms4([t1, t2, t3, t4]) => {
                terms[0] = t1;
                terms[1] = t2;
                terms[2] = t3;
                terms[3] = t4;
                len = 4;
            }
            ConstLC::Terms5([t1, t2, t3, t4, t5]) => {
                terms[0] = t1;
                terms[1] = t2;
                terms[2] = t3;
                terms[3] = t4;
                terms[4] = t5;
                len = 5;
            }
            ConstLC::Terms1Const([t1], k) => {
                terms[0] = t1;
                len = 1;
                c = k;
            }
            ConstLC::Terms2Const([t1, t2], k) => {
                terms[0] = t1;
                terms[1] = t2;
                len = 2;
                c = k;
            }
            ConstLC::Terms3Const([t1, t2, t3], k) => {
                terms[0] = t1;
                terms[1] = t2;
                terms[2] = t3;
                len = 3;
                c = k;
            }
            ConstLC::Terms4Const([t1, t2, t3, t4], k) => {
                terms[0] = t1;
                terms[1] = t2;
                terms[2] = t3;
                terms[3] = t4;
                len = 4;
                c = k;
            }
            ConstLC::Terms5Const([t1, t2, t3, t4, t5], k) => {
                terms[0] = t1;
                terms[1] = t2;
                terms[2] = t3;
                terms[3] = t4;
                terms[4] = t5;
                len = 5;
                c = k;
            }
        }
        (terms, len, c)
    }

    /// Compose a ConstLC from (terms, len, const)
    const fn compose(terms: &[ConstTerm; 5], len: usize, c: i128) -> ConstLC {
        match (len, c) {
            (0, 0) => ConstLC::Zero,
            (0, k) => ConstLC::Const(k),
            (1, 0) => ConstLC::Terms1([terms[0]]),
            (2, 0) => ConstLC::Terms2([terms[0], terms[1]]),
            (3, 0) => ConstLC::Terms3([terms[0], terms[1], terms[2]]),
            (4, 0) => ConstLC::Terms4([terms[0], terms[1], terms[2], terms[3]]),
            (5, 0) => ConstLC::Terms5([terms[0], terms[1], terms[2], terms[3], terms[4]]),
            (1, k) => ConstLC::Terms1Const([terms[0]], k),
            (2, k) => ConstLC::Terms2Const([terms[0], terms[1]], k),
            (3, k) => ConstLC::Terms3Const([terms[0], terms[1], terms[2]], k),
            (4, k) => ConstLC::Terms4Const([terms[0], terms[1], terms[2], terms[3]], k),
            (5, k) => ConstLC::Terms5Const([terms[0], terms[1], terms[2], terms[3], terms[4]], k),
            _ => ConstLC::Zero,
        }
    }

    /// Multiply this ConstLC by a constant
    pub const fn mul_by_const(self, multiplier: i128) -> ConstLC {
        match self {
            ConstLC::Zero => ConstLC::Zero,
            ConstLC::Const(c) => ConstLC::Const(c * multiplier),
            ConstLC::Terms1([t1]) => {
                ConstLC::Terms1([ConstTerm::new(t1.input_index, t1.coeff * multiplier)])
            }
            ConstLC::Terms2([t1, t2]) => ConstLC::Terms2([
                ConstTerm::new(t1.input_index, t1.coeff * multiplier),
                ConstTerm::new(t2.input_index, t2.coeff * multiplier),
            ]),
            ConstLC::Terms3([t1, t2, t3]) => ConstLC::Terms3([
                ConstTerm::new(t1.input_index, t1.coeff * multiplier),
                ConstTerm::new(t2.input_index, t2.coeff * multiplier),
                ConstTerm::new(t3.input_index, t3.coeff * multiplier),
            ]),
            ConstLC::Terms4([t1, t2, t3, t4]) => ConstLC::Terms4([
                ConstTerm::new(t1.input_index, t1.coeff * multiplier),
                ConstTerm::new(t2.input_index, t2.coeff * multiplier),
                ConstTerm::new(t3.input_index, t3.coeff * multiplier),
                ConstTerm::new(t4.input_index, t4.coeff * multiplier),
            ]),
            ConstLC::Terms5([t1, t2, t3, t4, t5]) => ConstLC::Terms5([
                ConstTerm::new(t1.input_index, t1.coeff * multiplier),
                ConstTerm::new(t2.input_index, t2.coeff * multiplier),
                ConstTerm::new(t3.input_index, t3.coeff * multiplier),
                ConstTerm::new(t4.input_index, t4.coeff * multiplier),
                ConstTerm::new(t5.input_index, t5.coeff * multiplier),
            ]),
            ConstLC::Terms1Const([t1], c) => ConstLC::Terms1Const(
                [ConstTerm::new(t1.input_index, t1.coeff * multiplier)],
                c * multiplier,
            ),
            ConstLC::Terms2Const([t1, t2], c) => ConstLC::Terms2Const(
                [
                    ConstTerm::new(t1.input_index, t1.coeff * multiplier),
                    ConstTerm::new(t2.input_index, t2.coeff * multiplier),
                ],
                c * multiplier,
            ),
            ConstLC::Terms3Const([t1, t2, t3], c) => ConstLC::Terms3Const(
                [
                    ConstTerm::new(t1.input_index, t1.coeff * multiplier),
                    ConstTerm::new(t2.input_index, t2.coeff * multiplier),
                    ConstTerm::new(t3.input_index, t3.coeff * multiplier),
                ],
                c * multiplier,
            ),
            ConstLC::Terms4Const([t1, t2, t3, t4], c) => ConstLC::Terms4Const(
                [
                    ConstTerm::new(t1.input_index, t1.coeff * multiplier),
                    ConstTerm::new(t2.input_index, t2.coeff * multiplier),
                    ConstTerm::new(t3.input_index, t3.coeff * multiplier),
                    ConstTerm::new(t4.input_index, t4.coeff * multiplier),
                ],
                c * multiplier,
            ),
            ConstLC::Terms5Const([t1, t2, t3, t4, t5], c) => ConstLC::Terms5Const(
                [
                    ConstTerm::new(t1.input_index, t1.coeff * multiplier),
                    ConstTerm::new(t2.input_index, t2.coeff * multiplier),
                    ConstTerm::new(t3.input_index, t3.coeff * multiplier),
                    ConstTerm::new(t4.input_index, t4.coeff * multiplier),
                    ConstTerm::new(t5.input_index, t5.coeff * multiplier),
                ],
                c * multiplier,
            ),
        }
    }
}

// =============================================================================
// CONST LC HELPERS AND MACROS
// =============================================================================

/// Public const helpers used by macros
pub const fn lc_add(a: ConstLC, b: ConstLC) -> ConstLC {
    match a.add_const_lc(b) {
        Some(lc) => lc,
        None => ConstLC::zero(),
    }
}

pub const fn lc_sub(a: ConstLC, b: ConstLC) -> ConstLC {
    match a.add_const_lc(b.mul_by_const(-1)) {
        Some(lc) => lc,
        None => ConstLC::zero(),
    }
}

pub const fn lc_mul_const(a: ConstLC, k: i128) -> ConstLC {
    a.mul_by_const(k)
}

pub const fn lc_from_input(inp: JoltR1CSInputs) -> ConstLC {
    ConstLC::single_term(inp.idx(), 1)
}

pub const fn lc_from_input_with_coeff(inp: JoltR1CSInputs, coeff: i128) -> ConstLC {
    ConstLC::single_term(inp.idx(), coeff)
}

pub const fn lc_const(k: i128) -> ConstLC {
    ConstLC::constant(k)
}

/// lc!: parse a linear combination with +, -, and literal * expr
/// Examples:
/// - lc!({ JoltR1CSInputs::UnexpandedPC } + { 4i128 } - { 2 * JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed) })
/// - lc!({ JoltR1CSInputs::LeftInstructionInput })
#[macro_export]
macro_rules! lc {
    // Entry points: normalize to accumulator form
    ( { $k:literal * $e:expr } $( $rest:tt )* ) => {
        $crate::lc!(@acc $crate::zkvm::r1cs::const_constraints::lc_from_input_with_coeff($e, $k) ; $( $rest )* )
    };
    ( { $k:literal } $( $rest:tt )* ) => {
        $crate::lc!(@acc $crate::zkvm::r1cs::const_constraints::lc_const($k) ; $( $rest )* )
    };
    ( { $e:expr } $( $rest:tt )* ) => {
        $crate::lc!(@acc $crate::zkvm::r1cs::const_constraints::lc_from_input($e) ; $( $rest )* )
    };

    // Accumulator folding rules
    (@acc $acc:expr ; + { $k:literal * $e:expr } $( $rest:tt )* ) => {
        $crate::lc!(@acc $crate::zkvm::r1cs::const_constraints::lc_add($acc, $crate::zkvm::r1cs::const_constraints::lc_from_input_with_coeff($e, $k)) ; $( $rest )* )
    };
    (@acc $acc:expr ; - { $k:literal * $e:expr } $( $rest:tt )* ) => {
        $crate::lc!(@acc $crate::zkvm::r1cs::const_constraints::lc_sub($acc, $crate::zkvm::r1cs::const_constraints::lc_from_input_with_coeff($e, $k)) ; $( $rest )* )
    };
    (@acc $acc:expr ; + { $k:literal } $( $rest:tt )* ) => {
        $crate::lc!(@acc $crate::zkvm::r1cs::const_constraints::lc_add($acc, $crate::zkvm::r1cs::const_constraints::lc_const($k)) ; $( $rest )* )
    };
    (@acc $acc:expr ; - { $k:literal } $( $rest:tt )* ) => {
        $crate::lc!(@acc $crate::zkvm::r1cs::const_constraints::lc_sub($acc, $crate::zkvm::r1cs::const_constraints::lc_const($k)) ; $( $rest )* )
    };
    (@acc $acc:expr ; + { $e:expr } $( $rest:tt )* ) => {
        $crate::lc!(@acc $crate::zkvm::r1cs::const_constraints::lc_add($acc, $crate::zkvm::r1cs::const_constraints::lc_from_input($e)) ; $( $rest )* )
    };
    (@acc $acc:expr ; - { $e:expr } $( $rest:tt )* ) => {
        $crate::lc!(@acc $crate::zkvm::r1cs::const_constraints::lc_sub($acc, $crate::zkvm::r1cs::const_constraints::lc_from_input($e)) ; $( $rest )* )
    };

    // End of input
    (@acc $acc:expr ; ) => { $acc };
}

/// r1cs_eq_conditional!: verbose, condition-first equality constraint
/// Usage: r1cs_eq_conditional!(if { COND } => { LEFT } == { RIGHT });
#[macro_export]
macro_rules! r1cs_eq_conditional {
    (if { $($cond:tt)* } => ( $($left:tt)* ) == ( $($right:tt)* ) ) => {{
        $crate::zkvm::r1cs::const_constraints::cs_constrain_eq_conditional_lc(
            $crate::lc!($($cond)*),
            $crate::lc!($($left)*),
            $crate::lc!($($right)*),
        )
    }};
}

/// r1cs_if_else!: verbose if-then-else with explicit result
/// Usage: r1cs_if_else!(if { COND } => { TRUE } else { FALSE } => { RESULT });
#[macro_export]
macro_rules! r1cs_if_else {
    (if { $($cond:tt)* } => ( $($tval:tt)* ) else ( $($fval:tt)* ) => ( $($result:tt)* ) ) => {{
        $crate::zkvm::r1cs::const_constraints::cs_constrain_if_else_lc(
            $crate::lc!($($cond)*),
            $crate::lc!($($tval)*),
            $crate::lc!($($fval)*),
            $crate::lc!($($result)*),
        )
    }};
}

/// r1cs_prod!: product constraint
/// Usage: r1cs_prod!({ LEFT } * { RIGHT } == { RESULT });
#[macro_export]
macro_rules! r1cs_prod {
    ( ( $($left:tt)* ) * ( $($right:tt)* ) == ( $($result:tt)* ) ) => {{
        $crate::zkvm::r1cs::const_constraints::cs_constrain_prod_lc(
            $crate::lc!($($left)*),
            $crate::lc!($($right)*),
            $crate::lc!($($result)*),
        )
    }};
}

/// A single R1CS constraint row
#[derive(Clone, Copy, Debug)]
pub struct ConstraintConst {
    pub a: ConstLC,
    pub b: ConstLC,
    pub c: ConstLC,
}

impl ConstraintConst {
    pub const fn new(a: ConstLC, b: ConstLC, c: ConstLC) -> Self {
        Self { a, b, c }
    }
}

/// Number of uniform R1CS constraints
pub const NUM_R1CS_CONSTRAINTS: usize = 28;

// =============================================================================
// CONSTRAINT BUILDER FUNCTIONS
// =============================================================================
/// Creates: condition * (left - right) == 0
pub const fn cs_constrain_eq_conditional_lc(
    condition: ConstLC,
    left: ConstLC,
    right: ConstLC,
) -> ConstraintConst {
    ConstraintConst::new(
        condition,
        match left.add_const_lc(right.mul_by_const(-1)) {
            Some(b) => b,
            None => ConstLC::zero(),
        },
        ConstLC::zero(),
    )
}

/// Creates: left * right == result
pub const fn cs_constrain_prod_lc(
    left: ConstLC,
    right: ConstLC,
    result: ConstLC,
) -> ConstraintConst {
    ConstraintConst::new(left, right, result)
}

/// Creates: condition * (true_val - false_val) == (result - false_val)
pub const fn cs_constrain_if_else_lc(
    condition: ConstLC,
    true_val: ConstLC,
    false_val: ConstLC,
    result: ConstLC,
) -> ConstraintConst {
    ConstraintConst::new(
        condition,
        match true_val.add_const_lc(false_val.mul_by_const(-1)) {
            Some(b) => b,
            None => ConstLC::zero(),
        },
        match result.add_const_lc(false_val.mul_by_const(-1)) {
            Some(c) => c,
            None => ConstLC::zero(),
        },
    )
}

/// Static table of all 28 R1CS uniform constraints
/// Each constraint corresponds to one call to constrain_* in JoltRV32IMConstraints::uniform_constraints
pub static UNIFORM_ROWS: [ConstraintConst; NUM_R1CS_CONSTRAINTS] = [
    // if LeftOperandIsRs1Value { assert!(LeftInstructionInput == Rs1Value) }
    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsRs1Value) } }
        => ( { JoltR1CSInputs::LeftInstructionInput } ) == ( { JoltR1CSInputs::Rs1Value } )
    ),
    // if LeftOperandIsPC { assert!(LeftInstructionInput == UnexpandedPC) }
    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsPC) } }
        => ( { JoltR1CSInputs::LeftInstructionInput } ) == ( { JoltR1CSInputs::UnexpandedPC } )
    ),
    // if !(LeftOperandIsRs1Value || LeftOperandIsPC)  {
    //     assert!(LeftInstructionInput == 0)
    // }
    // Note that LeftOperandIsRs1Value and LeftOperandIsPC are mutually exclusive flags
    r1cs_eq_conditional!(
        if { { 1i128 } - { JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsRs1Value) } - { JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsPC) } }
        => ( { JoltR1CSInputs::LeftInstructionInput } ) == ( { 0i128 } )
    ),
    // if RightOperandIsRs2Value { assert!(RightInstructionInput == Rs2Value) }
    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsRs2Value) } }
        => ( { JoltR1CSInputs::RightInstructionInput } ) == ( { JoltR1CSInputs::Rs2Value } )
    ),
    // if RightOperandIsImm { assert!(RightInstructionInput == Imm) }
    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsImm) } }
        => ( { JoltR1CSInputs::RightInstructionInput } ) == ( { JoltR1CSInputs::Imm } )
    ),
    // if !(RightOperandIsRs2Value || RightOperandIsImm)  {
    //     assert!(RightInstructionInput == 0)
    // }
    // Note that RightOperandIsRs2Value and RightOperandIsImm are mutually exclusive flags
    r1cs_eq_conditional!(
        if { { 1i128 } - { JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsRs2Value) } - { JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsImm) } }
        => ( { JoltR1CSInputs::RightInstructionInput } ) == ( { 0i128 } )
    ),
    // if Load || Store {
    //     assert!(RamAddress == Rs1Value + Imm)
    // } else {
    //     assert!(RamAddress == 0)
    // }
    r1cs_if_else!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::Load) } + { JoltR1CSInputs::OpFlags(CircuitFlags::Store) } }
        => ( { JoltR1CSInputs::Rs1Value } + { JoltR1CSInputs::Imm } )
        else ( { 0i128 } )
        => ( { JoltR1CSInputs::RamAddress } )
    ),
    // if Load {
    //     assert!(RamReadValue == RamWriteValue)
    // }
    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::Load) } }
        => ( { JoltR1CSInputs::RamReadValue } ) == ( { JoltR1CSInputs::RamWriteValue } )
    ),
    // if Load {
    //     assert!(RamReadValue == RdWriteValue)
    // }
    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::Load) } }
        => ( { JoltR1CSInputs::RamReadValue } ) == ( { JoltR1CSInputs::RdWriteValue } )
    ),
    // if Store {
    //     assert!(Rs2Value == RamWriteValue)
    // }
    r1cs_eq_conditional!(
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
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands) } + { JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands) } + { JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands) } }
        => ( { 0i128 } )
        else ( { JoltR1CSInputs::LeftInstructionInput } )
        => ( { JoltR1CSInputs::LeftLookupOperand } )
    ),
    // If AddOperands {
    //     assert!(RightLookupOperand == LeftInstructionInput + RightInstructionInput)
    // }
    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands) } }
        => ( { JoltR1CSInputs::RightLookupOperand } ) == ( { JoltR1CSInputs::LeftInstructionInput } + { JoltR1CSInputs::RightInstructionInput } )
    ),
    // If SubtractOperands {
    //     assert!(RightLookupOperand == LeftInstructionInput - RightInstructionInput)
    // }
    // Converts from unsigned to twos-complement representation
    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands) } }
        => ( { JoltR1CSInputs::RightLookupOperand } ) == ( { JoltR1CSInputs::LeftInstructionInput } - { JoltR1CSInputs::RightInstructionInput } + { 0x10000000000000000i128 } )
    ),
    // if MultiplyOperands {
    //     assert!(RightLookupOperand == Rs1Value * Rs2Value)
    // }
    r1cs_prod!(
        ({ JoltR1CSInputs::RightInstructionInput }) * ({ JoltR1CSInputs::LeftInstructionInput })
            == ({ JoltR1CSInputs::Product })
    ),
    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands) } }
        => ( { JoltR1CSInputs::RightLookupOperand } ) == ( { JoltR1CSInputs::Product } )
    ),
    // if !(AddOperands || SubtractOperands || MultiplyOperands || Advice) {
    //     assert!(RightLookupOperand == RightInstructionInput)
    // }
    // Arbitrary untrusted advice goes in right lookup operand
    r1cs_eq_conditional!(
        if { { 1i128 } - { JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands) } - { JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands) } - { JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands) } - { JoltR1CSInputs::OpFlags(CircuitFlags::Advice) } }
        => ( { JoltR1CSInputs::RightLookupOperand } ) == ( { JoltR1CSInputs::RightInstructionInput } )
    ),
    // if Assert {
    //     assert!(LookupOutput == 1)
    // }
    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::Assert) } }
        => ( { JoltR1CSInputs::LookupOutput } ) == ( { 1i128 } )
    ),
    // if Rd != 0 && WriteLookupOutputToRD {
    //     assert!(RdWriteValue == LookupOutput)
    // }
    r1cs_prod!(
        ({ JoltR1CSInputs::Rd })
            * ({ JoltR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToRD) })
            == ({ JoltR1CSInputs::WriteLookupOutputToRD })
    ),
    r1cs_eq_conditional!(
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
        ({ JoltR1CSInputs::Rd }) * ({ JoltR1CSInputs::OpFlags(CircuitFlags::Jump) })
            == ({ JoltR1CSInputs::WritePCtoRD })
    ),
    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::WritePCtoRD } }
        => ( { JoltR1CSInputs::RdWriteValue } ) == ( { JoltR1CSInputs::UnexpandedPC } + { 4i128 } - { 2 * JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed) } )
    ),
    // if Jump && !NextIsNoop {
    //     assert!(NextUnexpandedPC == LookupOutput)
    // }
    r1cs_prod!(
        ({ JoltR1CSInputs::OpFlags(CircuitFlags::Jump) })
            * ({ 1i128 } - { JoltR1CSInputs::NextIsNoop })
            == ({ JoltR1CSInputs::ShouldJump })
    ),
    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::ShouldJump } }
        => ( { JoltR1CSInputs::NextUnexpandedPC } ) == ( { JoltR1CSInputs::LookupOutput } )
    ),
    // if Branch && LookupOutput {
    //     assert!(NextUnexpandedPC == UnexpandedPC + Imm)
    // }
    r1cs_prod!(
        ({ JoltR1CSInputs::OpFlags(CircuitFlags::Branch) }) * ({ JoltR1CSInputs::LookupOutput })
            == ({ JoltR1CSInputs::ShouldBranch })
    ),
    r1cs_eq_conditional!(
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
        ({ JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed) })
            * ({ JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC) })
            == ({ JoltR1CSInputs::CompressedDoNotUpdateUnexpPC })
    ),
    r1cs_eq_conditional!(
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
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::InlineSequenceInstruction) } }
        => ( { JoltR1CSInputs::NextPC } ) == ( { JoltR1CSInputs::PC } + { 1i128 } )
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ground_truth_validation() {
        use super::super::builder::R1CSBuilder;
        use super::super::constraints::{JoltRV32IMConstraints, R1CSConstraints};
        use ark_bn254::Fr as F;

        // Generate ground truth constraints using the original dynamic system
        let mut builder = R1CSBuilder::new();
        <JoltRV32IMConstraints as R1CSConstraints<F>>::uniform_constraints(&mut builder);
        let ground_truth_constraints = builder.get_constraints();

        // Verify we have the expected number of constraints
        assert_eq!(
            ground_truth_constraints.len(),
            NUM_R1CS_CONSTRAINTS,
            "Number of constraints mismatch! Expected {}, got {}",
            NUM_R1CS_CONSTRAINTS,
            ground_truth_constraints.len()
        );

        // Compare each const constraint with its ground truth equivalent
        for (i, (const_constraint, ground_truth)) in UNIFORM_ROWS
            .iter()
            .zip(ground_truth_constraints.iter())
            .enumerate()
        {
            println!("Checking constraint {}", i);

            // Convert const constraint to dynamic for comparison and compare
            let converted = const_constraint_to_dynamic_for_test(const_constraint);
            compare_constraints(&converted, ground_truth, i);
        }
    }

    /// Test-specific conversion function that always works regardless of feature flags
    fn const_constraint_to_dynamic_for_test(
        const_constraint: &ConstraintConst,
    ) -> super::super::builder::Constraint {
        use super::super::builder::Constraint;
        use super::super::ops::{Term, Variable, LC};

        fn const_lc_to_dynamic_lc_for_test(const_lc: &ConstLC) -> LC {
            let mut terms = Vec::new();

            // Add variable terms
            for i in 0..const_lc.num_terms() {
                if let Some(term) = const_lc.term(i) {
                    terms.push(Term(Variable::Input(term.input_index), term.coeff));
                }
            }

            // Add constant term if present
            if let Some(const_val) = const_lc.const_term() {
                terms.push(Term(Variable::Constant, const_val));
            }

            LC::new(terms)
        }

        Constraint {
            a: const_lc_to_dynamic_lc_for_test(&const_constraint.a),
            b: const_lc_to_dynamic_lc_for_test(&const_constraint.b),
            c: const_lc_to_dynamic_lc_for_test(&const_constraint.c),
        }
    }

    // Helper function to compare two constraints for equality
    fn compare_constraints(
        converted: &super::super::builder::Constraint,
        ground_truth: &super::super::builder::Constraint,
        constraint_index: usize,
    ) {
        use super::super::ops::{Term, Variable};

        // Helper to sort terms for comparison (ground truth might have different ordering)
        // Also filter out zero coefficient terms since they don't affect the constraint
        fn sort_and_filter_terms(terms: &[Term]) -> Vec<Term> {
            let mut sorted: Vec<Term> = terms.iter()
                .filter(|term| term.1 != 0)  // Filter out zero coefficients
                .copied()
                .collect();
            sorted.sort_by_key(|term| match term.0 {
                Variable::Input(idx) => (0, idx, term.1),
                Variable::Constant => (1, 0, term.1),
            });
            sorted
        }

        let converted_a = sort_and_filter_terms(converted.a.terms());
        let ground_truth_a = sort_and_filter_terms(ground_truth.a.terms());
        let converted_b = sort_and_filter_terms(converted.b.terms());
        let ground_truth_b = sort_and_filter_terms(ground_truth.b.terms());
        let converted_c = sort_and_filter_terms(converted.c.terms());
        let ground_truth_c = sort_and_filter_terms(ground_truth.c.terms());

        assert_eq!(
            converted_a, ground_truth_a,
            "Constraint {} A terms mismatch!\nConverted: {:?}\nGround truth: {:?}",
            constraint_index, converted_a, ground_truth_a
        );

        assert_eq!(
            converted_b, ground_truth_b,
            "Constraint {} B terms mismatch!\nConverted: {:?}\nGround truth: {:?}",
            constraint_index, converted_b, ground_truth_b
        );

        assert_eq!(
            converted_c, ground_truth_c,
            "Constraint {} C terms mismatch!\nConverted: {:?}\nGround truth: {:?}",
            constraint_index, converted_c, ground_truth_c
        );
    }
}
