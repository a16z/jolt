//! Computation traits for multilinear polynomials.
//!
//! These traits define the two core operations on multilinear polynomials
//! without coupling to data layout or storage. Concrete types like
//! [`Polynomial<F>`](crate::Polynomial) implement whichever traits they support,
//! and downstream code (commitment schemes, sumcheck) is generic over these
//! interfaces — enabling different backends (CPU, GPU) behind the same API.

use jolt_field::Field;

/// Multilinear polynomial evaluation at an arbitrary point.
///
/// Any multilinear polynomial $f: \mathbb{F}^n \to \mathbb{F}$ is uniquely
/// determined by its $2^n$ evaluations on the Boolean hypercube. This trait
/// exposes point evaluation and dimensional metadata without prescribing how
/// the evaluations are stored.
pub trait MultilinearEvaluation<F: Field>: Send + Sync {
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
}

/// In-place variable binding — the core sumcheck operation.
///
/// Fixes the first variable to `scalar`, halving the evaluation table:
/// $$g(x_2, \ldots, x_n) = (1 - s) \cdot f(0, x_2, \ldots, x_n) + s \cdot f(1, x_2, \ldots, x_n)$$
///
/// After calling `bind`, `num_vars` decreases by 1 and `len` halves.
pub trait MultilinearBinding<F: Field>: Send + Sync {
    fn bind(&mut self, scalar: F);
}
