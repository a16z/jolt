//! Core trait for multilinear polynomial abstractions.

use std::borrow::Cow;

use jolt_field::Field;

use crate::DensePolynomial;

/// A multilinear polynomial over `F` in `num_vars` variables,
/// represented by its evaluations over the Boolean hypercube $\{0,1\}^n$.
///
/// Any multilinear polynomial $f: \mathbb{F}^n \to \mathbb{F}$ is uniquely
/// determined by its $2^n$ evaluations on the Boolean hypercube. This trait
/// provides a common interface for accessing those evaluations and performing
/// standard operations (evaluation, variable binding).
pub trait MultilinearPolynomial<F: Field>: Send + Sync {
    /// Number of variables $n$. The polynomial has $2^n$ evaluations.
    fn num_vars(&self) -> usize;

    /// Number of evaluations, equal to $2^n$.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Evaluates the polynomial at `point` $\in \mathbb{F}^n$ using the
    /// multilinear extension formula:
    /// $$f(r) = \sum_{x \in \{0,1\}^n} f(x) \cdot \widetilde{eq}(x, r)$$
    fn evaluate(&self, point: &[F]) -> F;

    /// Fixes the first variable to `scalar`, producing a polynomial in $n-1$ variables:
    /// $$g(x_2, \ldots, x_n) = (1 - s) \cdot f(0, x_2, \ldots, x_n) + s \cdot f(1, x_2, \ldots, x_n)$$
    fn bind(&self, scalar: F) -> DensePolynomial<F>;

    /// Returns all $2^n$ evaluations over the Boolean hypercube.
    ///
    /// Returns a borrowed slice when the underlying storage is already
    /// field elements, or an owned vector when conversion is required.
    fn evaluations(&self) -> Cow<'_, [F]>;
}
