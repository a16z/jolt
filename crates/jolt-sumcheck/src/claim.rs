//! Sumcheck claim: the public statement that the protocol proves.

use jolt_field::FieldCore;

pub use jolt_openings::EvaluationClaim;

/// Round count and degree bound for a sumcheck instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckShape {
    pub num_vars: usize,
    pub degree: usize,
}

impl SumcheckShape {
    /// Construct a sumcheck shape.
    ///
    /// # Panics
    ///
    /// Panics if `degree == 0`.
    pub fn new(num_vars: usize, degree: usize) -> Self {
        assert!(
            degree >= 1,
            "sumcheck round polynomial must have degree >= 1, got {degree}"
        );
        Self { num_vars, degree }
    }
}

impl<F: FieldCore> From<&SumcheckClaim<F>> for SumcheckShape {
    fn from(claim: &SumcheckClaim<F>) -> Self {
        Self {
            num_vars: claim.num_vars,
            degree: claim.degree,
        }
    }
}

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
pub struct SumcheckClaim<F: FieldCore> {
    /// Number of Boolean variables in the summation.
    pub num_vars: usize,
    /// Maximum degree of each round polynomial.
    pub degree: usize,
    /// The claimed value of the sum $\sum_{x \in \{0,1\}^n} g(x)$.
    pub claimed_sum: F,
}

impl<F: FieldCore> SumcheckClaim<F> {
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
