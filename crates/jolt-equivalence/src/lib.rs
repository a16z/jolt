//! Cross-system equivalence testing between jolt-core and jolt-zkvm.
//!
//! Provides [`StageTrace`] and [`ProtocolTrace`] types for normalizing
//! per-stage protocol data from either proving system, plus comparison
//! logic that reports the first mismatch with full context.
//!
//! Both systems are run with deterministic mock transcripts (same seed),
//! which forces identical challenges and therefore identical polynomial
//! arithmetic. Every intermediate value becomes directly comparable.

use std::fmt::Debug;

/// Per-stage comparison data extracted from either proving system.
///
/// With identical mock-transcript challenges and identical witnesses,
/// every field here must match between jolt-core and jolt-zkvm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageTrace<F: Debug + Clone + PartialEq + Eq> {
    /// Number of sumcheck rounds in this stage.
    pub num_rounds: usize,
    /// Degree of each round polynomial (constant within a stage).
    pub poly_degree: usize,
    /// Full coefficients of each round polynomial: `[round_idx][coeff_idx]`.
    pub round_poly_coeffs: Vec<Vec<F>>,
    /// Opening evaluations emitted after the sumcheck rounds.
    pub evals: Vec<F>,
}

/// Complete protocol trace across all sumcheck stages.
#[derive(Debug, Clone)]
pub struct ProtocolTrace<F: Debug + Clone + PartialEq + Eq> {
    /// One [`StageTrace`] per sumcheck stage, in execution order.
    pub stages: Vec<StageTrace<F>>,
}

impl<F: Debug + Clone + PartialEq + Eq> ProtocolTrace<F> {
    /// Panics with a detailed message at the first point of divergence.
    pub fn assert_equivalent(&self, other: &Self) {
        assert_eq!(
            self.stages.len(),
            other.stages.len(),
            "stage count: self={} other={}",
            self.stages.len(),
            other.stages.len(),
        );
        for (i, (a, b)) in self.stages.iter().zip(&other.stages).enumerate() {
            assert_eq!(a.num_rounds, b.num_rounds, "stage {i}: num_rounds");
            assert_eq!(a.poly_degree, b.poly_degree, "stage {i}: poly_degree");
            assert_eq!(
                a.round_poly_coeffs.len(),
                b.round_poly_coeffs.len(),
                "stage {i}: round count in coeffs",
            );
            for (j, (pa, pb)) in a
                .round_poly_coeffs
                .iter()
                .zip(&b.round_poly_coeffs)
                .enumerate()
            {
                assert_eq!(pa.len(), pb.len(), "stage {i} round {j}: coeff count");
                for (k, (ca, cb)) in pa.iter().zip(pb).enumerate() {
                    assert_eq!(*ca, *cb, "stage {i} round {j} coeff {k}");
                }
            }
            assert_eq!(a.evals.len(), b.evals.len(), "stage {i}: eval count");
            for (j, (ea, eb)) in a.evals.iter().zip(&b.evals).enumerate() {
                assert_eq!(*ea, *eb, "stage {i} eval {j}");
            }
        }
    }
}
