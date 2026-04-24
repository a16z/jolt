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

impl<F: Field> SumcheckClaim<F> {
    /// Construct a sumcheck claim.
    ///
    /// # Panics
    ///
    /// Panics if `degree == 0`. Sumcheck round polynomials must have
    /// degree ≥ 1; a constant round poly is meaningless.
    pub fn new(num_vars: usize, degree: usize, claimed_sum: F) -> Self {
        assert!(
            degree >= 1,
            "sumcheck round polynomial must have degree >= 1, got {degree}"
        );
        Self {
            num_vars,
            degree,
            claimed_sum,
        }
    }
}

/// Oracle evaluation claim produced by a successful sumcheck reduction.
///
/// Sumcheck reduces `∑_{x ∈ {0,1}^n} g(x) = C` to a single query
/// `g(r) = v` at a Fiat-Shamir-derived point `r`. The caller MUST
/// discharge this claim against the polynomial oracle (opening proof,
/// BlindFold, etc.) to retain soundness — sumcheck alone does not
/// verify `v` against any commitment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluationClaim<F: Field> {
    /// Challenge point `r = (r_1, ..., r_n)`.
    pub point: Vec<F>,
    /// Claimed evaluation `g(r) = v`.
    pub value: F,
}
