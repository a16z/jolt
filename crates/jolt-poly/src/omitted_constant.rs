use jolt_algebra::Field;
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
pub struct OmittedConstantPoly<F: Field> {
    coefficients: Vec<F>,
}

impl<F: Field> OmittedConstantPoly<F> {
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

    /// Recovers `q0` from an equality-factored round check
    /// `(1 - tau) q(0) + tau q(1) = claim`, as `claim - tau (q(1) - q0)`.
    pub fn constant_from_claim(&self, claim: F, tau: F) -> F {
        claim - tau * self.nonconstant_term_sum_at_one()
    }

    /// Returns `q(point)`, with `q0` recovered by
    /// [`constant_from_claim`](Self::constant_from_claim).
    pub fn evaluate_from_claim(&self, claim: F, tau: F, point: F) -> F {
        self.constant_from_claim(claim, tau) + self.evaluate_nonconstant_terms(point)
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
