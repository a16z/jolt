//! Compile-time constant R1CS constraints
//! 
//! This module provides a static, compile-time representation of R1CS constraints
//! to replace the dynamic constraint building in the prover's hot path.
//!
//! When the `const_r1cs` feature is enabled, the prover can use static constraint
//! tables instead of dynamic constraint building for better performance.
//!
//! ## Enhanced ConstExpr System
//! 
//! The module now includes an enhanced expression system that provides natural
//! syntax for building constraints:
//!
//! ### Basic Usage
//! 
//! ```rust
//! use jolt_core::zkvm::r1cs::const_constraints::*;
//! use jolt_core::zkvm::r1cs::inputs::JoltR1CSInputs;
//! use jolt_core::zkvm::instruction::CircuitFlags;
//! 
//! // Create expressions using const functions
//! let condition = flag(CircuitFlags::Load);
//! let left = input(JoltR1CSInputs::RamReadValue);  
//! let right = input(JoltR1CSInputs::RamWriteValue);
//! 
//! // Build constraints with natural syntax
//! let constraint = cs_constrain_eq_conditional_expr(condition, left, right);
//! // Creates: Load * (RamReadValue - RamWriteValue) == 0
//! ```
//!
//! ### Const Expression Helpers
//!
//! ```rust
//! // Add two flags: Load + Store
//! let load_or_store = add_2flags_expr(CircuitFlags::Load, CircuitFlags::Store);
//! 
//! // Subtract expressions: left - right  
//! let difference = sub_const_exprs(input(JoltR1CSInputs::Rs1Value), const_val(10));
//! 
//! // Common pattern: 1 - flag1 - flag2
//! let one_minus_flags = one_minus_2flags_expr(CircuitFlags::Load, CircuitFlags::Store);
//! ```
//!
//! ### Builder Functions
//!
//! The enhanced builder functions provide natural constraint construction:
//! - `cs_constrain_eq_conditional_expr(condition, left, right)` - conditional equality
//! - `cs_constrain_prod_expr(left, right, result)` - multiplication constraint  
//! - `cs_constrain_if_else_expr(condition, true_val, false_val, result)` - conditional assignment

use super::inputs::JoltR1CSInputs;
use crate::zkvm::instruction::CircuitFlags;

/// Helper functions for creating const expressions that mirror the original syntax
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

    pub const fn two_terms(idx1: usize, coeff1: i128, idx2: usize, coeff2: i128) -> Self {
        ConstLC::Terms2([
            ConstTerm::new(idx1, coeff1),
            ConstTerm::new(idx2, coeff2),
        ])
    }

    pub const fn three_terms(idx1: usize, coeff1: i128, idx2: usize, coeff2: i128, idx3: usize, coeff3: i128) -> Self {
        ConstLC::Terms3([
            ConstTerm::new(idx1, coeff1),
            ConstTerm::new(idx2, coeff2),
            ConstTerm::new(idx3, coeff3),
        ])
    }

    pub const fn four_terms(idx1: usize, coeff1: i128, idx2: usize, coeff2: i128, idx3: usize, coeff3: i128, idx4: usize, coeff4: i128) -> Self {
        ConstLC::Terms4([
            ConstTerm::new(idx1, coeff1),
            ConstTerm::new(idx2, coeff2),
            ConstTerm::new(idx3, coeff3),
            ConstTerm::new(idx4, coeff4),
        ])
    }

    pub const fn five_terms(idx1: usize, coeff1: i128, idx2: usize, coeff2: i128, idx3: usize, coeff3: i128, idx4: usize, coeff4: i128, idx5: usize, coeff5: i128) -> Self {
        ConstLC::Terms5([
            ConstTerm::new(idx1, coeff1),
            ConstTerm::new(idx2, coeff2),
            ConstTerm::new(idx3, coeff3),
            ConstTerm::new(idx4, coeff4),
            ConstTerm::new(idx5, coeff5),
        ])
    }

    pub const fn single_term_with_const(input_index: usize, coeff: i128, const_coeff: i128) -> Self {
        ConstLC::Terms1Const([ConstTerm::new(input_index, coeff)], const_coeff)
    }

    pub const fn two_terms_with_const(idx1: usize, coeff1: i128, idx2: usize, coeff2: i128, const_coeff: i128) -> Self {
        ConstLC::Terms2Const([
            ConstTerm::new(idx1, coeff1),
            ConstTerm::new(idx2, coeff2),
        ], const_coeff)
    }

    pub const fn three_terms_with_const(idx1: usize, coeff1: i128, idx2: usize, coeff2: i128, idx3: usize, coeff3: i128, const_coeff: i128) -> Self {
        ConstLC::Terms3Const([
            ConstTerm::new(idx1, coeff1),
            ConstTerm::new(idx2, coeff2),
            ConstTerm::new(idx3, coeff3),
        ], const_coeff)
    }

    pub const fn four_terms_with_const(idx1: usize, coeff1: i128, idx2: usize, coeff2: i128, idx3: usize, coeff3: i128, idx4: usize, coeff4: i128, const_coeff: i128) -> Self {
        ConstLC::Terms4Const([
            ConstTerm::new(idx1, coeff1),
            ConstTerm::new(idx2, coeff2),
            ConstTerm::new(idx3, coeff3),
            ConstTerm::new(idx4, coeff4),
        ], const_coeff)
    }

    pub const fn five_terms_with_const(idx1: usize, coeff1: i128, idx2: usize, coeff2: i128, idx3: usize, coeff3: i128, idx4: usize, coeff4: i128, idx5: usize, coeff5: i128, const_coeff: i128) -> Self {
        ConstLC::Terms5Const([
            ConstTerm::new(idx1, coeff1),
            ConstTerm::new(idx2, coeff2),
            ConstTerm::new(idx3, coeff3),
            ConstTerm::new(idx4, coeff4),
            ConstTerm::new(idx5, coeff5),
        ], const_coeff)
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
            ConstLC::Terms1(_) | ConstLC::Terms2(_) | ConstLC::Terms3(_) | ConstLC::Terms4(_) | ConstLC::Terms5(_) => None,
            ConstLC::Terms1Const(_, c) | ConstLC::Terms2Const(_, c) | ConstLC::Terms3Const(_, c) | ConstLC::Terms4Const(_, c) | ConstLC::Terms5Const(_, c) => Some(*c),
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
        let mut terms = [ConstTerm { input_index: 0, coeff: 0 }; 5];
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
            ConstLC::Terms1([t1]) => ConstLC::Terms1([ConstTerm::new(t1.input_index, t1.coeff * multiplier)]),
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
            ConstLC::Terms1Const([t1], c) => ConstLC::Terms1Const([
                ConstTerm::new(t1.input_index, t1.coeff * multiplier)
            ], c * multiplier),
            ConstLC::Terms2Const([t1, t2], c) => ConstLC::Terms2Const([
                ConstTerm::new(t1.input_index, t1.coeff * multiplier),
                ConstTerm::new(t2.input_index, t2.coeff * multiplier),
            ], c * multiplier),
            ConstLC::Terms3Const([t1, t2, t3], c) => ConstLC::Terms3Const([
                ConstTerm::new(t1.input_index, t1.coeff * multiplier),
                ConstTerm::new(t2.input_index, t2.coeff * multiplier),
                ConstTerm::new(t3.input_index, t3.coeff * multiplier),
            ], c * multiplier),
            ConstLC::Terms4Const([t1, t2, t3, t4], c) => ConstLC::Terms4Const([
                ConstTerm::new(t1.input_index, t1.coeff * multiplier),
                ConstTerm::new(t2.input_index, t2.coeff * multiplier),
                ConstTerm::new(t3.input_index, t3.coeff * multiplier),
                ConstTerm::new(t4.input_index, t4.coeff * multiplier),
            ], c * multiplier),
            ConstLC::Terms5Const([t1, t2, t3, t4, t5], c) => ConstLC::Terms5Const([
                ConstTerm::new(t1.input_index, t1.coeff * multiplier),
                ConstTerm::new(t2.input_index, t2.coeff * multiplier),
                ConstTerm::new(t3.input_index, t3.coeff * multiplier),
                ConstTerm::new(t4.input_index, t4.coeff * multiplier),
                ConstTerm::new(t5.input_index, t5.coeff * multiplier),
            ], c * multiplier),
        }
    }
}

// =============================================================================
// ENHANCED CONSTEXPR SYSTEM WITH TRAIT IMPLEMENTATIONS
// =============================================================================

/// Enhanced expression system that supports natural arithmetic syntax
/// Uses indices instead of Box for const compatibility
#[derive(Clone, Copy, Debug)]
pub enum ConstExpr {
    Const(i128),
    Input(JoltR1CSInputs),
}

impl ConstExpr {
    /// Flatten a ConstExpr tree into a ConstLC representation
    pub const fn flatten(self) -> ConstLC {
        match self {
            ConstExpr::Const(c) => ConstLC::constant(c),
            ConstExpr::Input(inp) => ConstLC::single_term(inp.idx(), 1),
        }
    }
}

// (Operator-based runtime expression system intentionally omitted for const-only design.)

// Automatic conversions for ergonomic use
impl From<i128> for ConstExpr {
    fn from(val: i128) -> Self {
        ConstExpr::Const(val)
    }
}

impl From<JoltR1CSInputs> for ConstExpr {
    fn from(inp: JoltR1CSInputs) -> Self {
        ConstExpr::Input(inp)
    }
}

// Const helper functions for static array compatibility
pub const fn const_val(v: i128) -> ConstExpr {
    ConstExpr::Const(v)
}

pub const fn input(inp: JoltR1CSInputs) -> ConstExpr {
    ConstExpr::Input(inp)
}

pub const fn flag(f: CircuitFlags) -> ConstExpr {
    ConstExpr::Input(JoltR1CSInputs::OpFlags(f))
}

/// Const helper to add two ConstExpr and return ConstLC
pub const fn add_const_exprs(left: ConstExpr, right: ConstExpr) -> ConstLC {
    let left_lc = left.flatten();
    let right_lc = right.flatten();
    match left_lc.add_const_lc(right_lc) {
        Some(result) => result,
        None => ConstLC::zero(), // Fallback when combination fails
    }
}

/// Const helper to subtract two ConstExpr and return ConstLC  
pub const fn sub_const_exprs(left: ConstExpr, right: ConstExpr) -> ConstLC {
    let left_lc = left.flatten();
    let right_lc = right.flatten().mul_by_const(-1);
    match left_lc.add_const_lc(right_lc) {
        Some(result) => result,
        None => ConstLC::zero(), // Fallback when combination fails
    }
}

/// Const helper to multiply ConstExpr by constant and return ConstLC
pub const fn mul_const_expr(expr: ConstExpr, constant: i128) -> ConstLC {
    expr.flatten().mul_by_const(constant)
}

pub const fn zero() -> ConstExpr {
    ConstExpr::Const(0)
}

// Helper functions for common patterns using ConstExpr
pub const fn one_minus_2flags_expr(f1: CircuitFlags, f2: CircuitFlags) -> ConstLC {
    // For now, use the existing helper that works directly with ConstLC
    one_minus_2flags(f1, f2)
}

pub const fn add_2flags_expr(f1: CircuitFlags, f2: CircuitFlags) -> ConstLC {
    add_const_exprs(flag(f1), flag(f2))
}

pub const fn add_2inputs_expr(inp1: JoltR1CSInputs, inp2: JoltR1CSInputs) -> ConstLC {
    add_const_exprs(input(inp1), input(inp2))
}



// =============================================================================
// CONST LC HELPERS AND MACROS
// =============================================================================

/// Public const helpers used by macros
pub const fn lc_add(a: ConstLC, b: ConstLC) -> ConstLC {
    match a.add_const_lc(b) { Some(lc) => lc, None => ConstLC::zero() }
}

pub const fn lc_sub(a: ConstLC, b: ConstLC) -> ConstLC {
    match a.add_const_lc(b.mul_by_const(-1)) { Some(lc) => lc, None => ConstLC::zero() }
}

pub const fn lc_mul_const(a: ConstLC, k: i128) -> ConstLC { a.mul_by_const(k) }

pub const fn lc_from_input(inp: JoltR1CSInputs) -> ConstLC {
    ConstLC::single_term(inp.idx(), 1)
}

pub const fn lc_from_input_with_coeff(inp: JoltR1CSInputs, coeff: i128) -> ConstLC {
    ConstLC::single_term(inp.idx(), coeff)
}

pub const fn lc_const(k: i128) -> ConstLC { ConstLC::constant(k) }

/// lc_simple!: parse a single atom into ConstLC without any recursion or +/- folding
#[macro_export]
macro_rules! lc_simple {
    ( { $k:literal * $e:expr } ) => { $crate::zkvm::r1cs::const_constraints::lc_from_input_with_coeff($e, $k) };
    ( { $k:literal } ) => { $crate::zkvm::r1cs::const_constraints::lc_const($k) };
    ( { $e:expr } ) => { $crate::zkvm::r1cs::const_constraints::lc_from_input($e) };
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
// ENHANCED BUILDER FUNCTIONS USING CONSTEXPR (NEW DEFAULT)
// =============================================================================

/// Enhanced cs.constrain_eq_conditional using ConstExpr
/// Creates: condition * (left - right) == 0
pub const fn cs_constrain_eq_conditional_expr(
    condition: ConstExpr,
    left: ConstExpr,
    right: ConstExpr,
) -> ConstraintConst {
    ConstraintConst::new(
        condition.flatten(),
        sub_const_exprs(left, right),
        ConstLC::zero(),
    )
}

/// Enhanced cs.constrain_prod using ConstExpr  
/// Creates: left * right == result
pub const fn cs_constrain_prod_expr(
    left: ConstExpr,
    right: ConstExpr,
    result: ConstExpr,
) -> ConstraintConst {
    ConstraintConst::new(
        left.flatten(),
        right.flatten(),
        result.flatten(),
    )
}

/// Enhanced cs.constrain_if_else using ConstExpr
/// Creates: condition * (true_val - false_val) == (result - false_val)
pub const fn cs_constrain_if_else_expr(
    condition: ConstExpr,
    true_val: ConstExpr,
    false_val: ConstExpr,
    result: ConstExpr,
) -> ConstraintConst {
    ConstraintConst::new(
        condition.flatten(),
        sub_const_exprs(true_val, false_val),
        sub_const_exprs(result, false_val),
    )
}

/// New extended-arity const builders that accept full ConstLCs
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
pub const fn cs_constrain_prod_lc(left: ConstLC, right: ConstLC, result: ConstLC) -> ConstraintConst {
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

// =============================================================================
// BUILDER FUNCTIONS THAT MIRROR THE ORIGINAL R1CSBuilder API (DIRECT INPUTS)
// =============================================================================

/// Mirror of cs.constrain_eq_conditional(condition, left, right)
/// Creates: condition * (left - right) == 0
pub const fn cs_constrain_eq_conditional(
    condition: JoltR1CSInputs,
    left: JoltR1CSInputs, 
    right: JoltR1CSInputs,
) -> ConstraintConst {
    ConstraintConst::new(
        ConstLC::single_term(condition.idx(), 1),                    // A = condition
        ConstLC::two_terms(left.idx(), 1, right.idx(), -1),         // B = left - right
        ConstLC::zero(),                                             // C = 0
    )
}

/// Mirror of cs.constrain_eq_conditional(condition, left, constant)
/// Creates: condition * (left - constant) == 0
pub const fn cs_constrain_eq_conditional_const(
    condition: JoltR1CSInputs,
    left: JoltR1CSInputs,
    constant: i128,
) -> ConstraintConst {
    ConstraintConst::new(
        ConstLC::single_term(condition.idx(), 1),                           // A = condition
        ConstLC::single_term_with_const(left.idx(), 1, -constant),         // B = left - constant
        ConstLC::zero(),                                                     // C = 0
    )
}

/// Mirror of cs.constrain_prod(left, right, result)
/// Creates: left * right == result
pub const fn cs_constrain_prod(
    left: JoltR1CSInputs,
    right: JoltR1CSInputs,
    result: JoltR1CSInputs,
) -> ConstraintConst {
    ConstraintConst::new(
        ConstLC::single_term(left.idx(), 1),     // A = left
        ConstLC::single_term(right.idx(), 1),    // B = right
        ConstLC::single_term(result.idx(), 1),   // C = result
    )
}

/// Mirror of cs.constrain_if_else(condition, true_val, false_val, result) where false_val is 0
/// Creates: condition * (true_val - 0) == (result - 0) => condition * true_val == result
pub const fn cs_constrain_if_else_zero(
    condition: JoltR1CSInputs,
    true_val: JoltR1CSInputs,
    result: JoltR1CSInputs,
) -> ConstraintConst {
    ConstraintConst::new(
        ConstLC::single_term(condition.idx(), 1),        // A = condition
        ConstLC::single_term(true_val.idx(), 1),         // B = true_val
        ConstLC::single_term(result.idx(), 1),           // C = result
    )
}

/// Mirror of cs.constrain_if_else(condition, true_val, false_val, result)
/// Creates: condition * (true_val - false_val) == (result - false_val)
pub const fn cs_constrain_if_else(
    condition: JoltR1CSInputs,
    true_val: JoltR1CSInputs,
    false_val: JoltR1CSInputs,
    result: JoltR1CSInputs,
) -> ConstraintConst {
    ConstraintConst::new(
        ConstLC::single_term(condition.idx(), 1),                                  // A = condition
        ConstLC::two_terms(true_val.idx(), 1, false_val.idx(), -1),              // B = true_val - false_val
        ConstLC::two_terms(result.idx(), 1, false_val.idx(), -1),                // C = result - false_val
    )
}

// =============================================================================
// HELPERS FOR COMPLEX EXPRESSIONS (like 1 - A - B - C)
// =============================================================================

/// Helper for: 1 - flag1 - flag2
pub const fn one_minus_2flags(flag1: CircuitFlags, flag2: CircuitFlags) -> ConstLC {
    ConstLC::two_terms_with_const(
        JoltR1CSInputs::OpFlags(flag1).idx(), -1, 
        JoltR1CSInputs::OpFlags(flag2).idx(), -1, 
        1
    )
}

/// Helper for: 1 - flag1 - flag2 - flag3 - flag4
pub const fn one_minus_4flags(flag1: CircuitFlags, flag2: CircuitFlags, flag3: CircuitFlags, flag4: CircuitFlags) -> ConstLC {
    ConstLC::four_terms_with_const(
        JoltR1CSInputs::OpFlags(flag1).idx(), -1,
        JoltR1CSInputs::OpFlags(flag2).idx(), -1,
        JoltR1CSInputs::OpFlags(flag3).idx(), -1,
        JoltR1CSInputs::OpFlags(flag4).idx(), -1,
        1
    )
}

/// Helper for: flag1 + flag2
pub const fn add_2flags(flag1: CircuitFlags, flag2: CircuitFlags) -> ConstLC {
    ConstLC::two_terms(
        JoltR1CSInputs::OpFlags(flag1).idx(), 1,
        JoltR1CSInputs::OpFlags(flag2).idx(), 1
    )
}

/// Helper for: flag1 + flag2 + flag3
pub const fn add_3flags(flag1: CircuitFlags, flag2: CircuitFlags, flag3: CircuitFlags) -> ConstLC {
    ConstLC::three_terms(
        JoltR1CSInputs::OpFlags(flag1).idx(), 1,
        JoltR1CSInputs::OpFlags(flag2).idx(), 1,
        JoltR1CSInputs::OpFlags(flag3).idx(), 1
    )
}

/// Helper for: input1 + input2
pub const fn add_2inputs(input1: JoltR1CSInputs, input2: JoltR1CSInputs) -> ConstLC {
    ConstLC::two_terms(input1.idx(), 1, input2.idx(), 1)
}

/// Helper for: input1 + input2 + input3
pub const fn add_3inputs(input1: JoltR1CSInputs, input2: JoltR1CSInputs, input3: JoltR1CSInputs) -> ConstLC {
    ConstLC::three_terms(input1.idx(), 1, input2.idx(), 1, input3.idx(), 1)
}

/// Helper for: input + constant
pub const fn input_plus_const(input: JoltR1CSInputs, constant: i128) -> ConstLC {
    ConstLC::single_term_with_const(input.idx(), 1, constant)
}

/// Helper for the complex PC update constraint
pub const fn complex_pc_update_expression(
    next_pc: JoltR1CSInputs,
    pc: JoltR1CSInputs,
    do_not_update: CircuitFlags,
    is_compressed: CircuitFlags,
    compressed_do_not_update: JoltR1CSInputs,
) -> ConstLC {
    ConstLC::five_terms_with_const(
        next_pc.idx(), 1,
        pc.idx(), -1,
        JoltR1CSInputs::OpFlags(do_not_update).idx(), 4,
        JoltR1CSInputs::OpFlags(is_compressed).idx(), 2,
        compressed_do_not_update.idx(), -2,
        -4
    )
}

/// Helper for: 1 - input1 - flag
pub const fn one_minus_input_and_flag(input: JoltR1CSInputs, flag: CircuitFlags) -> ConstLC {
    ConstLC::two_terms_with_const(
        input.idx(), -1,
        JoltR1CSInputs::OpFlags(flag).idx(), -1,
        1
    )
}

/// Helper for: 1 - flag
pub const fn one_minus_flag(flag: CircuitFlags) -> ConstLC {
    ConstLC::single_term_with_const(JoltR1CSInputs::OpFlags(flag).idx(), -1, 1)
}

/// Static table of all 28 R1CS uniform constraints
/// Each constraint corresponds to one call to constrain_* in JoltRV32IMConstraints::uniform_constraints
pub static UNIFORM_ROWS: [ConstraintConst; NUM_R1CS_CONSTRAINTS] = [
    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsRs1Value) } }
        => ( { JoltR1CSInputs::LeftInstructionInput } ) == ( { JoltR1CSInputs::Rs1Value } )
    ),

    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsPC) } }
        => ( { JoltR1CSInputs::LeftInstructionInput } ) == ( { JoltR1CSInputs::UnexpandedPC } )
    ),

    r1cs_eq_conditional!(
        if { { 1i128 } - { JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsRs1Value) } - { JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsPC) } }
        => ( { JoltR1CSInputs::LeftInstructionInput } ) == ( { 0i128 } )
    ),

    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsRs2Value) } }
        => ( { JoltR1CSInputs::RightInstructionInput } ) == ( { JoltR1CSInputs::Rs2Value } )
    ),

    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsImm) } }
        => ( { JoltR1CSInputs::RightInstructionInput } ) == ( { JoltR1CSInputs::Imm } )
    ),

    r1cs_eq_conditional!(
        if { { 1i128 } - { JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsRs2Value) } - { JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsImm) } }
        => ( { JoltR1CSInputs::RightInstructionInput } ) == ( { 0i128 } )
    ),

    r1cs_if_else!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::Load) } + { JoltR1CSInputs::OpFlags(CircuitFlags::Store) } }
        => ( { JoltR1CSInputs::Rs1Value } + { JoltR1CSInputs::Imm } )
        else ( { 0i128 } )
        => ( { JoltR1CSInputs::RamAddress } )
    ),

    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::Load) } }
        => ( { JoltR1CSInputs::RamReadValue } ) == ( { JoltR1CSInputs::RamWriteValue } )
    ),

    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::Load) } }
        => ( { JoltR1CSInputs::RamReadValue } ) == ( { JoltR1CSInputs::RdWriteValue } )
    ),

    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::Store) } }
        => ( { JoltR1CSInputs::Rs2Value } ) == ( { JoltR1CSInputs::RamWriteValue } )
    ),

    r1cs_if_else!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands) } + { JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands) } + { JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands) } }
        => ( { 0i128 } )
        else ( { JoltR1CSInputs::LeftInstructionInput } )
        => ( { JoltR1CSInputs::LeftLookupOperand } )
    ),

    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands) } }
        => ( { JoltR1CSInputs::RightLookupOperand } ) == ( { JoltR1CSInputs::LeftInstructionInput } + { JoltR1CSInputs::RightInstructionInput } )
    ),

    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands) } }
        => ( { JoltR1CSInputs::RightLookupOperand } ) == ( { JoltR1CSInputs::LeftInstructionInput } - { JoltR1CSInputs::RightInstructionInput } + { 0x10000000000000000i128 } )
    ),

    r1cs_prod!(
        ( { JoltR1CSInputs::RightInstructionInput } ) * ( { JoltR1CSInputs::LeftInstructionInput } ) == ( { JoltR1CSInputs::Product } )
    ),

    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands) } }
        => ( { JoltR1CSInputs::RightLookupOperand } ) == ( { JoltR1CSInputs::Product } )
    ),

    r1cs_eq_conditional!(
        if { { 1i128 } - { JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands) } - { JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands) } - { JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands) } - { JoltR1CSInputs::OpFlags(CircuitFlags::Advice) } }
        => ( { JoltR1CSInputs::RightLookupOperand } ) == ( { JoltR1CSInputs::RightInstructionInput } )
    ),

    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::Assert) } }
        => ( { JoltR1CSInputs::LookupOutput } ) == ( { 1i128 } )
    ),

    r1cs_prod!(
        ( { JoltR1CSInputs::Rd } ) * ( { JoltR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToRD) } ) == ( { JoltR1CSInputs::WriteLookupOutputToRD } )
    ),

    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::WriteLookupOutputToRD } }
        => ( { JoltR1CSInputs::RdWriteValue } ) == ( { JoltR1CSInputs::LookupOutput } )
    ),

    r1cs_prod!(
        ( { JoltR1CSInputs::Rd } ) * ( { JoltR1CSInputs::OpFlags(CircuitFlags::Jump) } ) == ( { JoltR1CSInputs::WritePCtoRD } )
    ),

    // cs.constrain_eq_conditional(JoltR1CSInputs::WritePCtoRD, JoltR1CSInputs::RdWriteValue, JoltR1CSInputs::UnexpandedPC + 4 - 2 * IsCompressed)
    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::WritePCtoRD } }
        => ( { JoltR1CSInputs::RdWriteValue } ) == ( { JoltR1CSInputs::UnexpandedPC } + { 4i128 } - { 2 * JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed) } )
    ),

    // cs.constrain_prod(Jump, 1 - NextIsNoop, ShouldJump)
    r1cs_prod!(
        ( { JoltR1CSInputs::OpFlags(CircuitFlags::Jump) } )
        * ( { 1i128 } - { JoltR1CSInputs::NextIsNoop } )
        == ( { JoltR1CSInputs::ShouldJump } )
    ),

    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::ShouldJump } }
        => ( { JoltR1CSInputs::NextUnexpandedPC } ) == ( { JoltR1CSInputs::LookupOutput } )
    ),

    r1cs_prod!(
        ( { JoltR1CSInputs::OpFlags(CircuitFlags::Branch) } ) * ( { JoltR1CSInputs::LookupOutput } ) == ( { JoltR1CSInputs::ShouldBranch } )
    ),

    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::ShouldBranch } }
        => ( { JoltR1CSInputs::NextUnexpandedPC } ) == ( { JoltR1CSInputs::UnexpandedPC } + { JoltR1CSInputs::Imm } )
    ),

    r1cs_prod!(
        ( { JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed) } ) * ( { JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC) } ) == ( { JoltR1CSInputs::CompressedDoNotUpdateUnexpPC } )
    ),

    r1cs_eq_conditional!(
        if { { 1i128 } - { JoltR1CSInputs::ShouldBranch } - { JoltR1CSInputs::OpFlags(CircuitFlags::Jump) } }
        => ( { JoltR1CSInputs::NextUnexpandedPC } )
           == ( { JoltR1CSInputs::UnexpandedPC } + { 4i128 }
                - { 4 * JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC) }
                - { 2 * JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed) }
                + { 2 * JoltR1CSInputs::CompressedDoNotUpdateUnexpPC } )
    ),

    r1cs_eq_conditional!(
        if { { JoltR1CSInputs::OpFlags(CircuitFlags::InlineSequenceInstruction) } }
        => ( { JoltR1CSInputs::NextPC } ) == ( { JoltR1CSInputs::PC } + { 1i128 } )
    ),
];

/// Convert a ConstLC to the dynamic LC format for compatibility with existing code
#[cfg(feature = "const_r1cs")]
pub fn const_lc_to_dynamic_lc(const_lc: &ConstLC) -> super::ops::LC {
    use super::ops::{LC, Term, Variable};
    
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

/// Convert a const constraint to the dynamic constraint format for compatibility
#[cfg(feature = "const_r1cs")]
pub fn const_constraint_to_dynamic(const_constraint: &ConstraintConst) -> super::builder::Constraint {
    use super::builder::Constraint;
    
    Constraint {
        a: const_lc_to_dynamic_lc(&const_constraint.a),
        b: const_lc_to_dynamic_lc(&const_constraint.b),
        c: const_lc_to_dynamic_lc(&const_constraint.c),
    }
}

/// Get all uniform constraints in dynamic format for compatibility
#[cfg(feature = "const_r1cs")]
pub fn get_uniform_constraints_dynamic() -> Vec<super::builder::Constraint> {
    UNIFORM_ROWS.iter()
        .map(const_constraint_to_dynamic)
        .collect()
}

/// Get a specific constraint in dynamic format
#[cfg(feature = "const_r1cs")]
pub fn get_constraint_dynamic(index: usize) -> super::builder::Constraint {
    assert!(index < NUM_R1CS_CONSTRAINTS, "Constraint index {} out of bounds", index);
    const_constraint_to_dynamic(&UNIFORM_ROWS[index])
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::inputs::JoltR1CSInputs;
    use crate::zkvm::instruction::CircuitFlags;

    #[test]
    fn test_const_expr_system() {
        // Test basic ConstExpr creation
        let const_expr = const_val(42);
        let input_expr = input(JoltR1CSInputs::Rs1Value);
        let flag_expr = flag(CircuitFlags::Load);
        
        // Test flattening
        assert_eq!(const_expr.flatten(), ConstLC::constant(42));
        assert_eq!(input_expr.flatten(), ConstLC::single_term(JoltR1CSInputs::Rs1Value.idx(), 1));
        assert_eq!(flag_expr.flatten(), ConstLC::single_term(JoltR1CSInputs::OpFlags(CircuitFlags::Load).idx(), 1));
        
        // Test const operations
        let addition = add_const_exprs(const_val(10), const_val(5));
        assert_eq!(addition, ConstLC::constant(15));
        
        let subtraction = sub_const_exprs(const_val(10), const_val(3));
        assert_eq!(subtraction, ConstLC::constant(7));
        
        // Test flag addition helper
        let flags_sum = add_2flags_expr(CircuitFlags::Load, CircuitFlags::Store);
        assert_eq!(flags_sum.num_terms(), 2);
        
        // Test input addition helper  
        let inputs_sum = add_2inputs_expr(JoltR1CSInputs::Rs1Value, JoltR1CSInputs::Rs2Value);
        assert_eq!(inputs_sum.num_terms(), 2);
    }

    #[test]
    fn test_enhanced_builder_functions() {
        // Test enhanced constraint builders
        let condition_expr = flag(CircuitFlags::Load);
        let left_expr = input(JoltR1CSInputs::RamReadValue);
        let right_expr = input(JoltR1CSInputs::RamWriteValue);
        
        let constraint = cs_constrain_eq_conditional_expr(condition_expr, left_expr, right_expr);
        
        // Should create: Load * (RamReadValue - RamWriteValue) == 0
        assert_eq!(constraint.a.num_terms(), 1);
        assert_eq!(constraint.a.term(0), Some(ConstTerm::new(JoltR1CSInputs::OpFlags(CircuitFlags::Load).idx(), 1)));
        
        assert_eq!(constraint.b.num_terms(), 2);
        assert_eq!(constraint.b.term(0), Some(ConstTerm::new(JoltR1CSInputs::RamReadValue.idx(), 1)));
        assert_eq!(constraint.b.term(1), Some(ConstTerm::new(JoltR1CSInputs::RamWriteValue.idx(), -1)));
        
        assert_eq!(constraint.c, ConstLC::zero());
    }

    #[test]
    fn test_const_lc_basic_operations() {
        // Test zero LC
        let zero = ConstLC::zero();
        assert_eq!(zero.num_terms(), 0);
        assert_eq!(zero.const_term(), None);

        // Test single term
        let single = ConstLC::single_term(5, 10);
        assert_eq!(single.num_terms(), 1);
        assert_eq!(single.term(0), Some(ConstTerm::new(5, 10)));
        assert_eq!(single.const_term(), None);

        // Test constant
        let const_lc = ConstLC::constant(42);
        assert_eq!(const_lc.num_terms(), 0);
        assert_eq!(const_lc.const_term(), Some(42));

        // Test two terms
        let two = ConstLC::two_terms(1, 3, 2, -5);
        assert_eq!(two.num_terms(), 2);
        assert_eq!(two.term(0), Some(ConstTerm::new(1, 3)));
        assert_eq!(two.term(1), Some(ConstTerm::new(2, -5)));
        assert_eq!(two.term(2), None);
    }

    #[test]
    fn test_uniform_constraints_basic() {
        // Test that we have the right number of constraints
        assert_eq!(UNIFORM_ROWS.len(), NUM_R1CS_CONSTRAINTS);
        assert_eq!(NUM_R1CS_CONSTRAINTS, 28);

        // Test first constraint structure
        let constraint_0 = &UNIFORM_ROWS[0];
        
        // Should be: LeftOperandIsRs1Value * (LeftInstructionInput - Rs1Value) == 0
        assert_eq!(constraint_0.a.num_terms(), 1);
        assert_eq!(constraint_0.a.term(0), Some(ConstTerm::new(JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsRs1Value).idx(), 1)));
        
        assert_eq!(constraint_0.b.num_terms(), 2);
        assert_eq!(constraint_0.b.term(0), Some(ConstTerm::new(JoltR1CSInputs::LeftInstructionInput.idx(), 1)));
        assert_eq!(constraint_0.b.term(1), Some(ConstTerm::new(JoltR1CSInputs::Rs1Value.idx(), -1)));
        
        assert_eq!(constraint_0.c.num_terms(), 0);

        // Test a multiplication constraint (constraint 14: RightInstructionInput * LeftInstructionInput == Product)
        let constraint_13 = &UNIFORM_ROWS[13];
        assert_eq!(constraint_13.a.num_terms(), 1);
        assert_eq!(constraint_13.a.term(0), Some(ConstTerm::new(JoltR1CSInputs::RightInstructionInput.idx(), 1)));
        
        assert_eq!(constraint_13.b.num_terms(), 1);
        assert_eq!(constraint_13.b.term(0), Some(ConstTerm::new(JoltR1CSInputs::LeftInstructionInput.idx(), 1)));
        
        assert_eq!(constraint_13.c.num_terms(), 1);
        assert_eq!(constraint_13.c.term(0), Some(ConstTerm::new(JoltR1CSInputs::Product.idx(), 1)));
    }

    #[test]
    #[cfg(feature = "const_r1cs")]
    fn test_const_to_dynamic_conversion() {
        // Test conversion of a simple constraint
        let const_constraint = &UNIFORM_ROWS[0];
        let dynamic_constraint = const_constraint_to_dynamic(const_constraint);
        
        // Verify the A term (should be single term with coefficient 1)
        assert_eq!(dynamic_constraint.a.num_terms(), 1);
        assert_eq!(dynamic_constraint.a.num_vars(), 1);
        
        // Verify the B term (should be two terms: +1 and -1)
        assert_eq!(dynamic_constraint.b.num_terms(), 2);
        assert_eq!(dynamic_constraint.b.num_vars(), 2);
        
        // Verify the C term (should be zero)
        assert_eq!(dynamic_constraint.c.num_terms(), 0);
        assert_eq!(dynamic_constraint.c.num_vars(), 0);
    }

    #[test]
    #[cfg(feature = "const_r1cs")]
    fn test_all_constraints_convert() {
        // Test that all 28 constraints can be converted without panicking
        let dynamic_constraints = get_uniform_constraints_dynamic();
        assert_eq!(dynamic_constraints.len(), NUM_R1CS_CONSTRAINTS);
        
        // Test individual access
        for i in 0..NUM_R1CS_CONSTRAINTS {
            let _constraint = get_constraint_dynamic(i);
        }
    }

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
        assert_eq!(ground_truth_constraints.len(), NUM_R1CS_CONSTRAINTS, 
                   "Number of constraints mismatch! Expected {}, got {}", 
                   NUM_R1CS_CONSTRAINTS, ground_truth_constraints.len());
        
        // Compare each const constraint with its ground truth equivalent
        for (i, (const_constraint, ground_truth)) in UNIFORM_ROWS.iter().zip(ground_truth_constraints.iter()).enumerate() {
            println!("Checking constraint {}", i);
            
            // Convert const constraint to dynamic for comparison and compare
            let converted = const_constraint_to_dynamic_for_test(const_constraint);
            compare_constraints(&converted, ground_truth, i);
        }
    }

    /// Test-specific conversion function that always works regardless of feature flags
    fn const_constraint_to_dynamic_for_test(const_constraint: &ConstraintConst) -> super::super::builder::Constraint {
        use super::super::builder::Constraint;
        use super::super::ops::{LC, Term, Variable};
        
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
    fn compare_constraints(converted: &super::super::builder::Constraint, ground_truth: &super::super::builder::Constraint, constraint_index: usize) {
        use super::super::ops::{Variable, Term};
        
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
        
        assert_eq!(converted_a, ground_truth_a,
                   "Constraint {} A terms mismatch!\nConverted: {:?}\nGround truth: {:?}", 
                   constraint_index, converted_a, ground_truth_a);
        
        assert_eq!(converted_b, ground_truth_b,
                   "Constraint {} B terms mismatch!\nConverted: {:?}\nGround truth: {:?}", 
                   constraint_index, converted_b, ground_truth_b);
        
        assert_eq!(converted_c, ground_truth_c,
                   "Constraint {} C terms mismatch!\nConverted: {:?}\nGround truth: {:?}", 
                   constraint_index, converted_c, ground_truth_c);
    }
}