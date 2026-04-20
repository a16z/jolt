//! Proof structures for single and batched sumcheck protocols.

use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use serde::{Deserialize, Serialize};

/// A sumcheck proof consisting of one univariate round polynomial per variable.
///
/// In round $i$ the prover sends a univariate polynomial $s_i(X)$ of degree
/// at most $d$ (the claim's degree bound). The verifier checks that
/// $s_i(0) + s_i(1)$ equals the running sum, then sets the next challenge
/// $r_i$ and updates the running sum to $s_i(r_i)$.
///
/// The proof is complete when all $n$ round polynomials have been sent;
/// the verifier is left with a single evaluation claim at the point
/// $(r_1, \ldots, r_n)$.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SumcheckProof<F: Field> {
    /// Round polynomials $s_1, \ldots, s_n$ in the order they were generated.
    pub round_polynomials: Vec<UnivariatePoly<F>>,
}
