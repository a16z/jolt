//! Sumcheck claim: the public statement that the protocol proves.

use jolt_field::Field;
/// A sumcheck claim asserting that
/// $\sum_{x \in \{0,1\}^n} g(x) = C$
/// where $g$ is a polynomial of individual degree at most `degree` in each variable.
///
/// # Fields
///
/// * `num_vars` -- number of Boolean variables $n$; the sum ranges over $2^n$ points.
/// * `degree` -- maximum total degree of the round polynomials the prover may send.
///   For a product of $k$ multilinear polynomials, `degree = k`.
/// * `claimed_sum` -- the value $C$ that the prover claims the sum equals.
#[derive(Clone, Debug)]
pub struct SumcheckClaim<F: Field> {
    /// Number of Boolean variables in the summation.
    pub num_vars: usize,
    /// Maximum degree of each round polynomial.
    pub degree: usize,
    /// The claimed value of the sum $\sum_{x \in \{0,1\}^n} g(x)$.
    pub claimed_sum: F,
}
