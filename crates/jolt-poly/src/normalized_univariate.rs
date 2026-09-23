//! Inner univariate polynomial with its constant coefficient omitted.

use jolt_field::Field;
use serde::{Deserialize, Serialize};

/// Stores `[q1, q2, ..., qd]` for an inner polynomial
/// `q(X) = q0 + q1 X + ... + qd X^d`.
///
/// The constant coefficient is supplied by the protocol using this polynomial.
/// Stored trailing zeros are retained because the payload length can determine
/// the expected round-message shape. An empty payload represents an inner
/// constant polynomial.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct NormalizedPoly<F: Field> {
    coefficients: Vec<F>,
}

impl<F: Field> NormalizedPoly<F> {
    /// Constructs from the stored nonconstant coefficients `[q1, ..., qd]`.
    pub fn new(coefficients: Vec<F>) -> Self {
        Self { coefficients }
    }

    /// Constructs from the full inner polynomial coefficients `[q0, ..., qd]`.
    pub fn from_q_coefficients(coefficients: Vec<F>) -> Self {
        Self::new(coefficients.into_iter().skip(1).collect())
    }

    /// Returns the stored nonconstant coefficients, including trailing zeros.
    pub fn coefficients(&self) -> &[F] {
        &self.coefficients
    }

    /// Consumes the representation and returns its stored coefficients.
    pub fn into_coefficients(self) -> Vec<F> {
        self.coefficients
    }

    /// Returns the stored degree bound of the inner polynomial.
    pub fn degree(&self) -> usize {
        self.coefficients.len()
    }

    /// Returns `q(1) - q0`.
    pub fn nonconstant_term_sum_at_one(&self) -> F {
        self.coefficients.iter().copied().sum()
    }

    /// Returns `q(point) - q0` using Horner's method.
    pub fn evaluate_nonconstant_terms(&self, point: F) -> F {
        self.coefficients
            .iter()
            .rev()
            .copied()
            .fold(F::zero(), |acc, coefficient| acc * point + coefficient)
            * point
    }
}
