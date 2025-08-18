//! Const-friendly R1CS linear combination operations
//!
//! This module provides compile-time constant operations for building R1CS constraints.
//! Unlike the legacy dynamic operations, these are designed to work with const contexts
//! and provide better performance in the prover's hot path.

use super::inputs::JoltR1CSInputs;
use crate::field::JoltField;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;

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

    #[cfg(test)]
    pub fn pretty_fmt(&self, f: &mut String) -> std::fmt::Result {
        use super::inputs::JoltR1CSInputs;
        use std::fmt::Write;

        if self.coeff == 1 {
            write!(f, "{:?}", JoltR1CSInputs::from_index(self.input_index))
        } else if self.coeff == -1 {
            write!(f, "-{:?}", JoltR1CSInputs::from_index(self.input_index))
        } else {
            write!(
                f,
                "{}⋅{:?}",
                self.coeff,
                JoltR1CSInputs::from_index(self.input_index)
            )
        }
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

    pub const fn num_terms(&self) -> usize {
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

    pub const fn const_term(&self) -> Option<i128> {
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
                    .mul_i128(term.coeff);
                result += value;
            }
        }

        // Add constant term if present
        if let Some(const_val) = self.const_term() {
            result += F::from_i128(const_val);
        }

        result
    }

    /// Compute Σ_j coeff_j * eq_ry[ j ] + c * eq_ry[ const_col ] without any dynamic iteration.
    /// Returns the column-side contribution (no row weight applied).
    #[inline(always)]
    pub fn dot_eq_ry<F: JoltField>(&self, eq_ry: &[F], const_col: usize) -> F {
        match self {
            ConstLC::Zero => F::zero(),
            ConstLC::Const(c) => eq_ry[const_col].mul_i128(*c),
            ConstLC::Terms1([t1]) => eq_ry[t1.input_index].mul_i128(t1.coeff),
            ConstLC::Terms2([t1, t2]) => {
                eq_ry[t1.input_index].mul_i128(t1.coeff) + eq_ry[t2.input_index].mul_i128(t2.coeff)
            }
            ConstLC::Terms3([t1, t2, t3]) => {
                eq_ry[t1.input_index].mul_i128(t1.coeff)
                    + eq_ry[t2.input_index].mul_i128(t2.coeff)
                    + eq_ry[t3.input_index].mul_i128(t3.coeff)
            }
            ConstLC::Terms4([t1, t2, t3, t4]) => {
                eq_ry[t1.input_index].mul_i128(t1.coeff)
                    + eq_ry[t2.input_index].mul_i128(t2.coeff)
                    + eq_ry[t3.input_index].mul_i128(t3.coeff)
                    + eq_ry[t4.input_index].mul_i128(t4.coeff)
            }
            ConstLC::Terms5([t1, t2, t3, t4, t5]) => {
                eq_ry[t1.input_index].mul_i128(t1.coeff)
                    + eq_ry[t2.input_index].mul_i128(t2.coeff)
                    + eq_ry[t3.input_index].mul_i128(t3.coeff)
                    + eq_ry[t4.input_index].mul_i128(t4.coeff)
                    + eq_ry[t5.input_index].mul_i128(t5.coeff)
            }
            ConstLC::Terms1Const([t1], c) => {
                eq_ry[t1.input_index].mul_i128(t1.coeff) + eq_ry[const_col].mul_i128(*c)
            }
            ConstLC::Terms2Const([t1, t2], c) => {
                eq_ry[t1.input_index].mul_i128(t1.coeff)
                    + eq_ry[t2.input_index].mul_i128(t2.coeff)
                    + eq_ry[const_col].mul_i128(*c)
            }
            ConstLC::Terms3Const([t1, t2, t3], c) => {
                eq_ry[t1.input_index].mul_i128(t1.coeff)
                    + eq_ry[t2.input_index].mul_i128(t2.coeff)
                    + eq_ry[t3.input_index].mul_i128(t3.coeff)
                    + eq_ry[const_col].mul_i128(*c)
            }
            ConstLC::Terms4Const([t1, t2, t3, t4], c) => {
                eq_ry[t1.input_index].mul_i128(t1.coeff)
                    + eq_ry[t2.input_index].mul_i128(t2.coeff)
                    + eq_ry[t3.input_index].mul_i128(t3.coeff)
                    + eq_ry[t4.input_index].mul_i128(t4.coeff)
                    + eq_ry[const_col].mul_i128(*c)
            }
            ConstLC::Terms5Const([t1, t2, t3, t4, t5], c) => {
                eq_ry[t1.input_index].mul_i128(t1.coeff)
                    + eq_ry[t2.input_index].mul_i128(t2.coeff)
                    + eq_ry[t3.input_index].mul_i128(t3.coeff)
                    + eq_ry[t4.input_index].mul_i128(t4.coeff)
                    + eq_ry[t5.input_index].mul_i128(t5.coeff)
                    + eq_ry[const_col].mul_i128(*c)
            }
        }
    }

    /// Serialize this ConstLC in canonical order to a byte vector
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
        let mut terms: Vec<(u32, i128)> = Vec::new();
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
            bytes.extend_from_slice(&coeff.to_le_bytes());
        }

        // constant term marker + value
        match self.const_term() {
            Some(c) => {
                bytes.push(1u8);
                bytes.extend_from_slice(&c.to_le_bytes());
            }
            None => bytes.push(0u8),
        }
    }

    /// Accumulate evaluations of this ConstLC into the evals vector
    /// Used for efficiently computing matrix-vector products
    #[inline]
    pub fn accumulate_evaluations<F: JoltField>(
        &self,
        evals: &mut [F],
        wr_scale: F,
        num_vars: usize,
    ) {
        self.for_each_term(|input_index, coeff| {
            evals[input_index] += wr_scale.mul_i128(coeff);
        });
        if let Some(c) = self.const_term() {
            evals[num_vars] += wr_scale.mul_i128(c);
        }
    }

    /// Iterate variable terms (input_index, coeff) without allocations.
    /// Order is not guaranteed except that it is consistent with this LC's internal storage.
    #[inline(always)]
    pub fn for_each_term(&self, mut f: impl FnMut(usize, i128)) {
        match self {
            ConstLC::Zero | ConstLC::Const(_) => {}
            ConstLC::Terms1([t1]) | ConstLC::Terms1Const([t1], _) => f(t1.input_index, t1.coeff),
            ConstLC::Terms2([t1, t2]) | ConstLC::Terms2Const([t1, t2], _) => {
                f(t1.input_index, t1.coeff);
                f(t2.input_index, t2.coeff);
            }
            ConstLC::Terms3([t1, t2, t3]) | ConstLC::Terms3Const([t1, t2, t3], _) => {
                f(t1.input_index, t1.coeff);
                f(t2.input_index, t2.coeff);
                f(t3.input_index, t3.coeff);
            }
            ConstLC::Terms4([t1, t2, t3, t4]) | ConstLC::Terms4Const([t1, t2, t3, t4], _) => {
                f(t1.input_index, t1.coeff);
                f(t2.input_index, t2.coeff);
                f(t3.input_index, t3.coeff);
                f(t4.input_index, t4.coeff);
            }
            ConstLC::Terms5([t1, t2, t3, t4, t5])
            | ConstLC::Terms5Const([t1, t2, t3, t4, t5], _) => {
                f(t1.input_index, t1.coeff);
                f(t2.input_index, t2.coeff);
                f(t3.input_index, t3.coeff);
                f(t4.input_index, t4.coeff);
                f(t5.input_index, t5.coeff);
            }
        }
    }

    #[cfg(test)]
    pub fn pretty_fmt(&self, f: &mut String) -> std::fmt::Result {
        use std::fmt::Write;

        match self {
            ConstLC::Zero => write!(f, "0"),
            ConstLC::Const(c) => write!(f, "{}", c),
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
                        if term.coeff == 0 {
                            continue;
                        }

                        if written_terms > 0 {
                            if term.coeff < 0 {
                                write!(f, " - ")?;
                                // Create a term with positive coefficient for display
                                let display_term = ConstTerm::new(term.input_index, -term.coeff);
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
                    if c != 0 {
                        if written_terms > 0 {
                            if c < 0 {
                                write!(f, " - {}", -c)?;
                            } else {
                                write!(f, " + {}", c)?;
                            }
                        } else {
                            write!(f, "{}", c)?;
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

    #[cfg(test)]
    pub fn assert_no_duplicate_terms(&self) {
        let mut input_indices = Vec::new();

        for i in 0..self.num_terms() {
            if let Some(term) = self.term(i) {
                if input_indices.contains(&term.input_index) {
                    panic!(
                        "Duplicate input index found in ConstLC: {}",
                        term.input_index
                    );
                } else {
                    input_indices.push(term.input_index);
                }
            }
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
        $crate::lc!(@acc $crate::zkvm::r1cs::ops::lc_from_input_with_coeff($e, $k) ; $( $rest )* )
    };
    ( { $k:literal } $( $rest:tt )* ) => {
        $crate::lc!(@acc $crate::zkvm::r1cs::ops::lc_const($k) ; $( $rest )* )
    };
    ( { $e:expr } $( $rest:tt )* ) => {
        $crate::lc!(@acc $crate::zkvm::r1cs::ops::lc_from_input($e) ; $( $rest )* )
    };

    // Accumulator folding rules
    (@acc $acc:expr ; + { $k:literal * $e:expr } $( $rest:tt )* ) => {
        $crate::lc!(@acc $crate::zkvm::r1cs::ops::lc_add($acc, $crate::zkvm::r1cs::ops::lc_from_input_with_coeff($e, $k)) ; $( $rest )* )
    };
    (@acc $acc:expr ; - { $k:literal * $e:expr } $( $rest:tt )* ) => {
        $crate::lc!(@acc $crate::zkvm::r1cs::ops::lc_sub($acc, $crate::zkvm::r1cs::ops::lc_from_input_with_coeff($e, $k)) ; $( $rest )* )
    };
    (@acc $acc:expr ; + { $k:literal } $( $rest:tt )* ) => {
        $crate::lc!(@acc $crate::zkvm::r1cs::ops::lc_add($acc, $crate::zkvm::r1cs::ops::lc_const($k)) ; $( $rest )* )
    };
    (@acc $acc:expr ; - { $k:literal } $( $rest:tt )* ) => {
        $crate::lc!(@acc $crate::zkvm::r1cs::ops::lc_sub($acc, $crate::zkvm::r1cs::ops::lc_const($k)) ; $( $rest )* )
    };
    (@acc $acc:expr ; + { $e:expr } $( $rest:tt )* ) => {
        $crate::lc!(@acc $crate::zkvm::r1cs::ops::lc_add($acc, $crate::zkvm::r1cs::ops::lc_from_input($e)) ; $( $rest )* )
    };
    (@acc $acc:expr ; - { $e:expr } $( $rest:tt )* ) => {
        $crate::lc!(@acc $crate::zkvm::r1cs::ops::lc_sub($acc, $crate::zkvm::r1cs::ops::lc_from_input($e)) ; $( $rest )* )
    };

    // End of input
    (@acc $acc:expr ; ) => { $acc };
}
