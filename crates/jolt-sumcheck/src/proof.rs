//! Proof structures for single and batched sumcheck protocols.

use crate::committed::CommittedSumcheckProof;
use jolt_poly::{CompressedPoly, UnivariatePoly};
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
pub struct ClearSumcheckProof<F: jolt_field::Field> {
    /// Round polynomials $s_1, \ldots, s_n$ in the order they were generated.
    pub round_polynomials: Vec<UnivariatePoly<F>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CompressedSumcheckProof<F: jolt_field::Field> {
    /// Boolean-hypercube round polynomials with the linear coefficient omitted.
    pub round_polynomials: Vec<CompressedPoly<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "C: Serialize", deserialize = "C: Deserialize<'de>"))]
pub enum SumcheckProof<F: jolt_field::Field, C> {
    Clear(CompressedSumcheckProof<F>),
    Committed(CommittedSumcheckProof<C>),
}

impl<F: jolt_field::Field, C> SumcheckProof<F, C> {
    pub fn as_clear(&self) -> Option<&CompressedSumcheckProof<F>> {
        match self {
            Self::Clear(proof) => Some(proof),
            Self::Committed(_) => None,
        }
    }

    pub fn as_committed(&self) -> Option<&CommittedSumcheckProof<C>> {
        match self {
            Self::Clear(_) => None,
            Self::Committed(proof) => Some(proof),
        }
    }
}
