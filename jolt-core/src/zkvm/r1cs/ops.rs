//! Const-friendly R1CS linear combination operations
//!
//! This module provides compile-time constant operations for building R1CS constraints.
//! Unlike the legacy dynamic operations, these are designed to work with const contexts
//! and provide better performance in the prover's hot path.

use super::inputs::JoltR1CSInputs;
use super::types::ConstantValue;
use crate::field::JoltField;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;

/// Helper for JoltR1CSInputs to get indices
impl JoltR1CSInputs {
    /// Convert this input to a usable index
    pub const fn idx(self) -> usize {
        self.to_index()
    }
}

/// A single term in a linear combination: (input_index, coefficient)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Term {
    pub input_index: usize,
    pub coeff: ConstantValue,
}

impl Term {
    /// Create a new term with given input index and coefficient.
    pub const fn new(input_index: usize, coeff: ConstantValue) -> Self {
        Self { input_index, coeff }
    }

    /// Create a new term with given input index and i128 coefficient.
    pub const fn new_i128(input_index: usize, coeff: i128) -> Self {
        Self {
            input_index,
            coeff: ConstantValue::from_i128(coeff),
        }
    }

    /// Format term for pretty printing (test only).
    #[cfg(test)]
    pub fn pretty_fmt(&self, f: &mut String) -> std::fmt::Result {
        use super::inputs::JoltR1CSInputs;
        use std::fmt::Write;

        let coeff_i128 = self.coeff.to_i128();
        if coeff_i128 == 1 {
            write!(f, "{:?}", JoltR1CSInputs::from_index(self.input_index))
        } else if coeff_i128 == -1 {
            write!(f, "-{:?}", JoltR1CSInputs::from_index(self.input_index))
        } else {
            write!(
                f,
                "{}⋅{:?}",
                coeff_i128,
                JoltR1CSInputs::from_index(self.input_index)
            )
        }
    }
}

/// Const-friendly linear combination enum that can hold 0-5 terms with an optional constant
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LC {
    Zero,
    Const(ConstantValue),
    Terms1([Term; 1]),
    Terms2([Term; 2]),
    Terms3([Term; 3]),
    Terms4([Term; 4]),
    Terms5([Term; 5]),
    Terms1Const([Term; 1], ConstantValue),
    Terms2Const([Term; 2], ConstantValue),
    Terms3Const([Term; 3], ConstantValue),
    Terms4Const([Term; 4], ConstantValue),
    Terms5Const([Term; 5], ConstantValue),
}

impl LC {
    pub const fn zero() -> Self {
        LC::Zero
    }

    pub const fn constant(value: ConstantValue) -> Self {
        LC::Const(value)
    }

    pub const fn constant_i128(value: i128) -> Self {
        LC::Const(ConstantValue::from_i128(value))
    }

    pub const fn single_term(input_index: usize, coeff: ConstantValue) -> Self {
        LC::Terms1([Term::new(input_index, coeff)])
    }

    pub const fn single_term_i128(input_index: usize, coeff: i128) -> Self {
        LC::Terms1([Term::new_i128(input_index, coeff)])
    }

    /// Create an LC from a single input with unit coefficient.
    pub const fn from_input(inp: JoltR1CSInputs) -> LC {
        LC::single_term(inp.idx(), ConstantValue::one())
    }

    /// Create an LC from a single input with explicit coefficient.
    pub const fn from_input_with_coeff(inp: JoltR1CSInputs, coeff: ConstantValue) -> LC {
        LC::single_term(inp.idx(), coeff)
    }

    /// Create an LC from a single input with explicit i128 coefficient.
    pub const fn from_input_with_coeff_i128(inp: JoltR1CSInputs, coeff: i128) -> LC {
        LC::single_term_i128(inp.idx(), coeff)
    }

    /// Create a constant LC.
    pub const fn from_const(k: ConstantValue) -> LC {
        LC::constant(k)
    }

    /// Create a constant LC from i128.
    pub const fn from_const_i128(k: i128) -> LC {
        LC::constant_i128(k)
    }

    // =========================
    // Introspection
    // =========================
    pub const fn num_terms(&self) -> usize {
        match self {
            LC::Zero | LC::Const(_) => 0,
            LC::Terms1(_) | LC::Terms1Const(_, _) => 1,
            LC::Terms2(_) | LC::Terms2Const(_, _) => 2,
            LC::Terms3(_) | LC::Terms3Const(_, _) => 3,
            LC::Terms4(_) | LC::Terms4Const(_, _) => 4,
            LC::Terms5(_) | LC::Terms5Const(_, _) => 5,
        }
    }

    pub fn term(&self, i: usize) -> Option<Term> {
        match self {
            LC::Zero | LC::Const(_) => None,
            LC::Terms1(terms) | LC::Terms1Const(terms, _) => terms.get(i).copied(),
            LC::Terms2(terms) | LC::Terms2Const(terms, _) => terms.get(i).copied(),
            LC::Terms3(terms) | LC::Terms3Const(terms, _) => terms.get(i).copied(),
            LC::Terms4(terms) | LC::Terms4Const(terms, _) => terms.get(i).copied(),
            LC::Terms5(terms) | LC::Terms5Const(terms, _) => terms.get(i).copied(),
        }
    }

    pub const fn const_term(&self) -> Option<ConstantValue> {
        match self {
            LC::Zero => None,
            LC::Const(c) => Some(*c),
            LC::Terms1(_) | LC::Terms2(_) | LC::Terms3(_) | LC::Terms4(_) | LC::Terms5(_) => None,
            LC::Terms1Const(_, c)
            | LC::Terms2Const(_, c)
            | LC::Terms3Const(_, c)
            | LC::Terms4Const(_, c)
            | LC::Terms5Const(_, c) => Some(*c),
        }
    }

    /// Capacity-checked addition. Returns None if term capacity would be exceeded.
    pub const fn checked_add(self, other: LC) -> Option<LC> {
        let (mut out_terms, mut out_len, mut out_const) = Self::decompose(self);
        let (rhs_terms, rhs_len, rhs_const) = Self::decompose(other);

        out_const = out_const.add(rhs_const);

        let mut i = 0usize;
        while i < rhs_len {
            let term = rhs_terms[i];

            let mut found = false;
            let mut j = 0usize;
            while j < out_len {
                if out_terms[j].input_index == term.input_index {
                    let new_coeff = out_terms[j].coeff.add(term.coeff);
                    if new_coeff.is_zero() {
                        if out_len > 0 {
                            out_len -= 1;
                            out_terms[j] = out_terms[out_len];
                        }
                    } else {
                        out_terms[j] = Term::new(term.input_index, new_coeff);
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

    /// Capacity-checked subtraction. Returns None if term capacity would be exceeded.
    pub const fn checked_sub(self, other: LC) -> Option<LC> {
        self.checked_add(other.mul_by_const(ConstantValue::from_i128(-1)))
    }

    /// Capacity-checked add-constant. Returns None if term capacity would be exceeded.
    pub const fn checked_add_const(self, k: ConstantValue) -> Option<LC> {
        self.checked_add(LC::Const(k))
    }

    /// Capacity-checked add-constant from i128. Returns None if term capacity would be exceeded.
    pub const fn checked_add_const_i128(self, k: i128) -> Option<LC> {
        self.checked_add(LC::Const(ConstantValue::from_i128(k)))
    }

    /// Addition that falls back to zero LC if capacity would be exceeded.
    pub const fn add_or_zero(self, other: LC) -> LC {
        match self.checked_add(other) {
            Some(lc) => lc,
            None => LC::zero(),
        }
    }

    /// Subtraction that falls back to zero LC if capacity would be exceeded.
    pub const fn sub_or_zero(self, other: LC) -> LC {
        match self.checked_sub(other) {
            Some(lc) => lc,
            None => LC::zero(),
        }
    }

    /// Add constant that falls back to zero LC if capacity would be exceeded.
    pub const fn add_const_or_zero(self, k: ConstantValue) -> LC {
        match self.checked_add_const(k) {
            Some(lc) => lc,
            None => LC::zero(),
        }
    }

    /// Add i128 constant that falls back to zero LC if capacity would be exceeded.
    pub const fn add_const_or_zero_i128(self, k: i128) -> LC {
        match self.checked_add_const_i128(k) {
            Some(lc) => lc,
            None => LC::zero(),
        }
    }

    /// Negate this LC (multiply by -1).
    pub const fn neg(self) -> LC {
        self.mul_by_const(ConstantValue::from_i128(-1))
    }

    /// Break a LC into (terms, len, const)
    const fn decompose(lc: LC) -> ([Term; 5], usize, ConstantValue) {
        let mut terms = [Term {
            input_index: 0,
            coeff: ConstantValue::zero(),
        }; 5];
        let mut len = 0usize;
        let mut c = ConstantValue::zero();
        match lc {
            LC::Zero => {}
            LC::Const(k) => {
                c = k;
            }
            LC::Terms1([t1]) => {
                terms[0] = t1;
                len = 1;
            }
            LC::Terms2([t1, t2]) => {
                terms[0] = t1;
                terms[1] = t2;
                len = 2;
            }
            LC::Terms3([t1, t2, t3]) => {
                terms[0] = t1;
                terms[1] = t2;
                terms[2] = t3;
                len = 3;
            }
            LC::Terms4([t1, t2, t3, t4]) => {
                terms[0] = t1;
                terms[1] = t2;
                terms[2] = t3;
                terms[3] = t4;
                len = 4;
            }
            LC::Terms5([t1, t2, t3, t4, t5]) => {
                terms[0] = t1;
                terms[1] = t2;
                terms[2] = t3;
                terms[3] = t4;
                terms[4] = t5;
                len = 5;
            }
            LC::Terms1Const([t1], k) => {
                terms[0] = t1;
                len = 1;
                c = k;
            }
            LC::Terms2Const([t1, t2], k) => {
                terms[0] = t1;
                terms[1] = t2;
                len = 2;
                c = k;
            }
            LC::Terms3Const([t1, t2, t3], k) => {
                terms[0] = t1;
                terms[1] = t2;
                terms[2] = t3;
                len = 3;
                c = k;
            }
            LC::Terms4Const([t1, t2, t3, t4], k) => {
                terms[0] = t1;
                terms[1] = t2;
                terms[2] = t3;
                terms[3] = t4;
                len = 4;
                c = k;
            }
            LC::Terms5Const([t1, t2, t3, t4, t5], k) => {
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

    /// Compose a LC from (terms, len, const)
    const fn compose(terms: &[Term; 5], len: usize, c: ConstantValue) -> LC {
        match (len, c.is_zero()) {
            (0, true) => LC::Zero,
            (0, false) => LC::Const(c),
            (1, true) => LC::Terms1([terms[0]]),
            (2, true) => LC::Terms2([terms[0], terms[1]]),
            (3, true) => LC::Terms3([terms[0], terms[1], terms[2]]),
            (4, true) => LC::Terms4([terms[0], terms[1], terms[2], terms[3]]),
            (5, true) => LC::Terms5([terms[0], terms[1], terms[2], terms[3], terms[4]]),
            (1, false) => LC::Terms1Const([terms[0]], c),
            (2, false) => LC::Terms2Const([terms[0], terms[1]], c),
            (3, false) => LC::Terms3Const([terms[0], terms[1], terms[2]], c),
            (4, false) => LC::Terms4Const([terms[0], terms[1], terms[2], terms[3]], c),
            (5, false) => LC::Terms5Const([terms[0], terms[1], terms[2], terms[3], terms[4]], c),
            _ => LC::Zero,
        }
    }

    /// Multiply this LC by a constant
    pub const fn mul_by_const(self, multiplier: ConstantValue) -> LC {
        match self {
            LC::Zero => LC::Zero,
            LC::Const(c) => LC::Const(c.mul(multiplier)),
            LC::Terms1([t1]) => LC::Terms1([Term::new(t1.input_index, t1.coeff.mul(multiplier))]),
            LC::Terms2([t1, t2]) => LC::Terms2([
                Term::new(t1.input_index, t1.coeff.mul(multiplier)),
                Term::new(t2.input_index, t2.coeff.mul(multiplier)),
            ]),
            LC::Terms3([t1, t2, t3]) => LC::Terms3([
                Term::new(t1.input_index, t1.coeff.mul(multiplier)),
                Term::new(t2.input_index, t2.coeff.mul(multiplier)),
                Term::new(t3.input_index, t3.coeff.mul(multiplier)),
            ]),
            LC::Terms4([t1, t2, t3, t4]) => LC::Terms4([
                Term::new(t1.input_index, t1.coeff.mul(multiplier)),
                Term::new(t2.input_index, t2.coeff.mul(multiplier)),
                Term::new(t3.input_index, t3.coeff.mul(multiplier)),
                Term::new(t4.input_index, t4.coeff.mul(multiplier)),
            ]),
            LC::Terms5([t1, t2, t3, t4, t5]) => LC::Terms5([
                Term::new(t1.input_index, t1.coeff.mul(multiplier)),
                Term::new(t2.input_index, t2.coeff.mul(multiplier)),
                Term::new(t3.input_index, t3.coeff.mul(multiplier)),
                Term::new(t4.input_index, t4.coeff.mul(multiplier)),
                Term::new(t5.input_index, t5.coeff.mul(multiplier)),
            ]),
            LC::Terms1Const([t1], c) => LC::Terms1Const(
                [Term::new(t1.input_index, t1.coeff.mul(multiplier))],
                c.mul(multiplier),
            ),
            LC::Terms2Const([t1, t2], c) => LC::Terms2Const(
                [
                    Term::new(t1.input_index, t1.coeff.mul(multiplier)),
                    Term::new(t2.input_index, t2.coeff.mul(multiplier)),
                ],
                c.mul(multiplier),
            ),
            LC::Terms3Const([t1, t2, t3], c) => LC::Terms3Const(
                [
                    Term::new(t1.input_index, t1.coeff.mul(multiplier)),
                    Term::new(t2.input_index, t2.coeff.mul(multiplier)),
                    Term::new(t3.input_index, t3.coeff.mul(multiplier)),
                ],
                c.mul(multiplier),
            ),
            LC::Terms4Const([t1, t2, t3, t4], c) => LC::Terms4Const(
                [
                    Term::new(t1.input_index, t1.coeff.mul(multiplier)),
                    Term::new(t2.input_index, t2.coeff.mul(multiplier)),
                    Term::new(t3.input_index, t3.coeff.mul(multiplier)),
                    Term::new(t4.input_index, t4.coeff.mul(multiplier)),
                ],
                c.mul(multiplier),
            ),
            LC::Terms5Const([t1, t2, t3, t4, t5], c) => LC::Terms5Const(
                [
                    Term::new(t1.input_index, t1.coeff.mul(multiplier)),
                    Term::new(t2.input_index, t2.coeff.mul(multiplier)),
                    Term::new(t3.input_index, t3.coeff.mul(multiplier)),
                    Term::new(t4.input_index, t4.coeff.mul(multiplier)),
                    Term::new(t5.input_index, t5.coeff.mul(multiplier)),
                ],
                c.mul(multiplier),
            ),
        }
    }

    /// Multiply this LC by an i128 constant
    pub const fn mul_by_const_i128(self, multiplier: i128) -> LC {
        self.mul_by_const(ConstantValue::from_i128(multiplier))
    }

    /// Evaluate this linear combination at a specific row in the witness polynomials
    #[inline]
    pub fn evaluate_row<F: JoltField>(
        &self,
        flattened_polynomials: &[MultilinearPolynomial<F>],
        row: usize,
    ) -> F {
        let mut result = F::zero();

        // Add variable terms
        for i in 0..self.num_terms() {
            if let Some(term) = self.term(i) {
                let value = flattened_polynomials[term.input_index]
                    .get_coeff(row)
                    .mul_constant_value(term.coeff);
                result += value;
            }
        }

        // Add constant term if present
        if let Some(const_val) = self.const_term() {
            result += F::from_constant_value(const_val);
        }

        result
    }

    /// Compute Σ_j coeff_j * eq_ry[ j ] + c * eq_ry[ const_col ] without any dynamic iteration.
    /// Returns the column-side contribution (no row weight applied).
    #[inline(always)]
    pub fn dot_eq_ry<F: JoltField>(&self, eq_ry: &[F], const_col: usize) -> F {
        match self {
            LC::Zero => F::zero(),
            LC::Const(c) => eq_ry[const_col].mul_constant_value(*c),
            LC::Terms1([t1]) => eq_ry[t1.input_index].mul_constant_value(t1.coeff),
            LC::Terms2([t1, t2]) => {
                eq_ry[t1.input_index].mul_constant_value(t1.coeff)
                    + eq_ry[t2.input_index].mul_constant_value(t2.coeff)
            }
            LC::Terms3([t1, t2, t3]) => {
                eq_ry[t1.input_index].mul_constant_value(t1.coeff)
                    + eq_ry[t2.input_index].mul_constant_value(t2.coeff)
                    + eq_ry[t3.input_index].mul_constant_value(t3.coeff)
            }
            LC::Terms4([t1, t2, t3, t4]) => {
                eq_ry[t1.input_index].mul_constant_value(t1.coeff)
                    + eq_ry[t2.input_index].mul_constant_value(t2.coeff)
                    + eq_ry[t3.input_index].mul_constant_value(t3.coeff)
                    + eq_ry[t4.input_index].mul_constant_value(t4.coeff)
            }
            LC::Terms5([t1, t2, t3, t4, t5]) => {
                eq_ry[t1.input_index].mul_constant_value(t1.coeff)
                    + eq_ry[t2.input_index].mul_constant_value(t2.coeff)
                    + eq_ry[t3.input_index].mul_constant_value(t3.coeff)
                    + eq_ry[t4.input_index].mul_constant_value(t4.coeff)
                    + eq_ry[t5.input_index].mul_constant_value(t5.coeff)
            }
            LC::Terms1Const([t1], c) => {
                eq_ry[t1.input_index].mul_constant_value(t1.coeff)
                    + eq_ry[const_col].mul_constant_value(*c)
            }
            LC::Terms2Const([t1, t2], c) => {
                eq_ry[t1.input_index].mul_constant_value(t1.coeff)
                    + eq_ry[t2.input_index].mul_constant_value(t2.coeff)
                    + eq_ry[const_col].mul_constant_value(*c)
            }
            LC::Terms3Const([t1, t2, t3], c) => {
                eq_ry[t1.input_index].mul_constant_value(t1.coeff)
                    + eq_ry[t2.input_index].mul_constant_value(t2.coeff)
                    + eq_ry[t3.input_index].mul_constant_value(t3.coeff)
                    + eq_ry[const_col].mul_constant_value(*c)
            }
            LC::Terms4Const([t1, t2, t3, t4], c) => {
                eq_ry[t1.input_index].mul_constant_value(t1.coeff)
                    + eq_ry[t2.input_index].mul_constant_value(t2.coeff)
                    + eq_ry[t3.input_index].mul_constant_value(t3.coeff)
                    + eq_ry[t4.input_index].mul_constant_value(t4.coeff)
                    + eq_ry[const_col].mul_constant_value(*c)
            }
            LC::Terms5Const([t1, t2, t3, t4, t5], c) => {
                eq_ry[t1.input_index].mul_constant_value(t1.coeff)
                    + eq_ry[t2.input_index].mul_constant_value(t2.coeff)
                    + eq_ry[t3.input_index].mul_constant_value(t3.coeff)
                    + eq_ry[t4.input_index].mul_constant_value(t4.coeff)
                    + eq_ry[t5.input_index].mul_constant_value(t5.coeff)
                    + eq_ry[const_col].mul_constant_value(*c)
            }
        }
    }

    /// Serialize this LC in canonical order to a byte vector
    /// Used for digest computation in key generation
    ///
    /// Format:
    /// - tag: u8 (identifies matrix A/B/C)
    /// - term_count: u8 (number of variable terms)
    /// - for each term: u32 input_index (big-endian), i128 coefficient (little-endian)
    /// - constant_marker: u8 (1 if constant term present, 0 otherwise)
    /// - if constant present: i128 constant_value (little-endian)
    ///
    /// Endianness choices:
    /// - Input indices use big-endian for natural sorting order in serialized form
    /// - Coefficients use little-endian for consistency with field element serialization
    pub fn serialize_canonical(&self, tag: u8, bytes: &mut Vec<u8>) {
        bytes.push(tag);

        // Collect variable terms and sort by input index for deterministic ordering
        let mut terms: Vec<(u32, ConstantValue)> = Vec::new();
        let term_count = self.num_terms();
        let mut i = 0usize;
        while i < term_count {
            if let Some(t) = self.term(i) {
                terms.push((t.input_index as u32, t.coeff));
            }
            i += 1;
        }
        // sort by input index for determinism
        terms.sort_by_key(|t| t.0);

        // write term count (u8) and each term as (u32 idx BE, i128 coeff LE)
        bytes.push(terms.len() as u8);
        for (idx, coeff) in terms.into_iter() {
            bytes.extend_from_slice(&idx.to_be_bytes());
            bytes.extend_from_slice(&coeff.to_i128().to_le_bytes());
        }

        // constant term marker + value
        match self.const_term() {
            Some(c) => {
                bytes.push(1u8);
                bytes.extend_from_slice(&c.to_i128().to_le_bytes());
            }
            None => bytes.push(0u8),
        }
    }

    /// Accumulate evaluations of this LC into the evals vector
    /// Used for efficiently computing matrix-vector products
    #[inline]
    pub fn accumulate_evaluations<F: JoltField>(
        &self,
        evals: &mut [F],
        wr_scale: F,
        num_vars: usize,
    ) {
        self.for_each_term(|input_index, coeff| {
            evals[input_index] += wr_scale.mul_constant_value(coeff);
        });
        if let Some(c) = self.const_term() {
            evals[num_vars] += wr_scale.mul_constant_value(c);
        }
    }

    /// Iterate variable terms (input_index, coeff) without allocations.
    /// Order is not guaranteed except that it is consistent with this LC's internal storage.
    #[inline(always)]
    pub fn for_each_term(&self, mut f: impl FnMut(usize, ConstantValue)) {
        match self {
            LC::Zero | LC::Const(_) => {}
            LC::Terms1([t1]) | LC::Terms1Const([t1], _) => f(t1.input_index, t1.coeff),
            LC::Terms2([t1, t2]) | LC::Terms2Const([t1, t2], _) => {
                f(t1.input_index, t1.coeff);
                f(t2.input_index, t2.coeff);
            }
            LC::Terms3([t1, t2, t3]) | LC::Terms3Const([t1, t2, t3], _) => {
                f(t1.input_index, t1.coeff);
                f(t2.input_index, t2.coeff);
                f(t3.input_index, t3.coeff);
            }
            LC::Terms4([t1, t2, t3, t4]) | LC::Terms4Const([t1, t2, t3, t4], _) => {
                f(t1.input_index, t1.coeff);
                f(t2.input_index, t2.coeff);
                f(t3.input_index, t3.coeff);
                f(t4.input_index, t4.coeff);
            }
            LC::Terms5([t1, t2, t3, t4, t5]) | LC::Terms5Const([t1, t2, t3, t4, t5], _) => {
                f(t1.input_index, t1.coeff);
                f(t2.input_index, t2.coeff);
                f(t3.input_index, t3.coeff);
                f(t4.input_index, t4.coeff);
                f(t5.input_index, t5.coeff);
            }
        }
    }

    /// Format LC for pretty printing (test only).
    #[cfg(test)]
    pub fn pretty_fmt(&self, f: &mut String) -> std::fmt::Result {
        use std::fmt::Write;

        match self {
            LC::Zero => write!(f, "0"),
            LC::Const(c) => write!(f, "{}", c.to_i128()),
            _ => {
                let num_terms = self.num_terms();
                let has_const = self.const_term().is_some();
                let total_parts = num_terms + if has_const { 1 } else { 0 };

                if total_parts > 1 {
                    write!(f, "(")?;
                }

                let mut written_terms = 0;

                // Write variable terms
                for i in 0..num_terms {
                    if let Some(term) = self.term(i) {
                        if term.coeff.to_i128() == 0 {
                            continue;
                        }

                        if written_terms > 0 {
                            if term.coeff.to_i128() < 0 {
                                write!(f, " - ")?;
                                // Create a term with positive coefficient for display
                                let display_term = Term::new(
                                    term.input_index,
                                    ConstantValue::from_i128(-term.coeff.to_i128()),
                                );
                                display_term.pretty_fmt(f)?;
                            } else {
                                write!(f, " + ")?;
                                term.pretty_fmt(f)?;
                            }
                        } else {
                            term.pretty_fmt(f)?;
                        }
                        written_terms += 1;
                    }
                }

                // Write constant term
                if let Some(c) = self.const_term() {
                    let c_i128 = c.to_i128();
                    if c_i128 != 0 {
                        if written_terms > 0 {
                            if c_i128 < 0 {
                                write!(f, " - {}", -c_i128)?;
                            } else {
                                write!(f, " + {c_i128}")?;
                            }
                        } else {
                            write!(f, "{c_i128}")?;
                        }
                    }
                }

                if total_parts > 1 {
                    write!(f, ")")?;
                }
                Ok(())
            }
        }
    }

    /// Assert this LC has no duplicate terms (test only).
    #[cfg(test)]
    pub fn assert_no_duplicate_terms(&self) {
        let mut input_indices = Vec::new();

        for i in 0..self.num_terms() {
            if let Some(term) = self.term(i) {
                if input_indices.contains(&term.input_index) {
                    panic!("Duplicate input index found in LC: {}", term.input_index);
                } else {
                    input_indices.push(term.input_index);
                }
            }
        }
    }
}

// =============================================================================
// LC MACRO
// =============================================================================
/// lc!: parse a linear combination with +, -, and literal * expr
/// Examples:
/// - lc!({ JoltR1CSInputs::UnexpandedPC } + { 4i128 } - { 2 * JoltR1CSInputs::OpFlags(CircuitFlags::IsCompressed) })
/// - lc!({ JoltR1CSInputs::LeftInstructionInput })
#[macro_export]
macro_rules! lc {
	// Entry points: normalize to accumulator form
	( { $k:literal * $e:expr } $( $rest:tt )* ) => {
		$crate::lc!(@acc $crate::zkvm::r1cs::ops::LC::from_input_with_coeff($e, $crate::zkvm::r1cs::types::ConstantValue::from_i128($k)) ; $( $rest )* )
	};
	( { $k:literal } $( $rest:tt )* ) => {
		$crate::lc!(@acc $crate::zkvm::r1cs::ops::LC::from_const($crate::zkvm::r1cs::types::ConstantValue::from_i128($k)) ; $( $rest )* )
	};
	( { $e:expr } $( $rest:tt )* ) => {
		$crate::lc!(@acc $crate::zkvm::r1cs::ops::LC::from_input($e) ; $( $rest )* )
	};

	// Accumulator folding rules
	(@acc $acc:expr ; + { $k:literal * $e:expr } $( $rest:tt )* ) => {
		$crate::lc!(@acc $acc.add_or_zero($crate::zkvm::r1cs::ops::LC::from_input_with_coeff($e, $crate::zkvm::r1cs::types::ConstantValue::from_i128($k))) ; $( $rest )* )
	};
	(@acc $acc:expr ; - { $k:literal * $e:expr } $( $rest:tt )* ) => {
		$crate::lc!(@acc $acc.sub_or_zero($crate::zkvm::r1cs::ops::LC::from_input_with_coeff($e, $crate::zkvm::r1cs::types::ConstantValue::from_i128($k))) ; $( $rest )* )
	};
	(@acc $acc:expr ; + { $k:literal } $( $rest:tt )* ) => {
		$crate::lc!(@acc $acc.add_const_or_zero($crate::zkvm::r1cs::types::ConstantValue::from_i128($k)) ; $( $rest )* )
	};
	(@acc $acc:expr ; - { $k:literal } $( $rest:tt )* ) => {
		$crate::lc!(@acc $acc.add_const_or_zero($crate::zkvm::r1cs::types::ConstantValue::from_i128(-$k)) ; $( $rest )* )
	};
	(@acc $acc:expr ; + { $e:expr } $( $rest:tt )* ) => {
		$crate::lc!(@acc $acc.add_or_zero($crate::zkvm::r1cs::ops::LC::from_input($e)) ; $( $rest )* )
	};
	(@acc $acc:expr ; - { $e:expr } $( $rest:tt )* ) => {
		$crate::lc!(@acc $acc.sub_or_zero($crate::zkvm::r1cs::ops::LC::from_input($e)) ; $( $rest )* )
	};

	// End of input
	(@acc $acc:expr ; ) => { $acc };
}
