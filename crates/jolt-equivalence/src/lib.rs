//! Cross-system equivalence testing between jolt-core and jolt-zkvm.
//!
//! Provides two complementary comparison strategies:
//!
//! - **Stage-level**: [`StageTrace`] and [`compare_stage`] normalize and compare
//!   per-stage sumcheck coefficients and evaluations.
//! - **Op-level**: [`CheckpointTranscript`] records every transcript operation
//!   and [`find_divergence`] pinpoints the exact op where two systems diverge.
//!
//! The checkpoint approach is strictly more precise: stage-level comparison
//! tells you "stage 2 diverges at round 5", while op-level comparison tells
//! you "the 47th transcript append diverges — it's the eq table construction
//! for the RamRW phase 1 kernel".

pub mod checkpoint;

use std::fmt;

pub use checkpoint::{
    assert_transcripts_match, find_divergence, CheckpointTranscript, TranscriptDivergence,
    TranscriptEvent,
};

/// Per-stage comparison data extracted from either proving system.
///
/// With identical mock-transcript challenges and identical witnesses,
/// every field here must match between jolt-core and jolt-zkvm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageTrace<F: fmt::Debug + Clone + PartialEq + Eq> {
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
pub struct ProtocolTrace<F: fmt::Debug + Clone + PartialEq + Eq> {
    pub stages: Vec<StageTrace<F>>,
}

impl<F: fmt::Debug + Clone + PartialEq + Eq> ProtocolTrace<F> {
    /// Panics with a detailed message at the first point of divergence.
    pub fn assert_equivalent(&self, other: &Self) {
        assert_eq!(
            self.stages.len(),
            other.stages.len(),
            "stage count: self={} other={}",
            self.stages.len(),
            other.stages.len(),
        );
        for i in 0..self.stages.len() {
            if let Err(e) = compare_stage(i, &self.stages[i], &other.stages[i]) {
                panic!("{e}");
            }
        }
    }
}

/// Describes the first point where two stage traces diverge.
pub struct StageDivergence {
    pub stage: usize,
    pub kind: DivergenceKind,
}

pub enum DivergenceKind {
    NumRounds {
        expected: usize,
        actual: usize,
    },
    PolyDegree {
        expected: usize,
        actual: usize,
    },
    RoundCount {
        expected: usize,
        actual: usize,
    },
    CoeffCount {
        round: usize,
        expected: usize,
        actual: usize,
    },
    Coefficient {
        round: usize,
        coeff: usize,
        expected: String,
        actual: String,
    },
    EvalCount {
        expected: usize,
        actual: usize,
    },
    Eval {
        index: usize,
        expected: String,
        actual: String,
    },
}

impl fmt::Display for StageDivergence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stage {}: ", self.stage + 1)?;
        match &self.kind {
            DivergenceKind::NumRounds { expected, actual } => {
                write!(f, "num_rounds: expected {expected}, got {actual}")
            }
            DivergenceKind::PolyDegree { expected, actual } => {
                write!(f, "poly_degree: expected {expected}, got {actual}")
            }
            DivergenceKind::RoundCount { expected, actual } => {
                write!(
                    f,
                    "round_poly count: expected {expected}, got {actual}"
                )
            }
            DivergenceKind::CoeffCount {
                round,
                expected,
                actual,
            } => write!(
                f,
                "round {round} coeff count: expected {expected}, got {actual}"
            ),
            DivergenceKind::Coefficient {
                round,
                coeff,
                expected,
                actual,
            } => write!(
                f,
                "round {round} coeff {coeff} mismatch:\n  expected: {expected}\n  actual:   {actual}"
            ),
            DivergenceKind::EvalCount { expected, actual } => {
                write!(f, "eval count: expected {expected}, got {actual}")
            }
            DivergenceKind::Eval {
                index,
                expected,
                actual,
            } => write!(
                f,
                "eval {index} mismatch:\n  expected: {expected}\n  actual:   {actual}"
            ),
        }
    }
}

impl fmt::Debug for StageDivergence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Compare a single stage from two protocol traces.
///
/// Returns `Ok(())` if all round polynomial coefficients and evaluations
/// match exactly, or `Err(StageDivergence)` at the first mismatch.
pub fn compare_stage<F: fmt::Debug + Clone + PartialEq + Eq>(
    stage: usize,
    reference: &StageTrace<F>,
    candidate: &StageTrace<F>,
) -> Result<(), StageDivergence> {
    if reference.num_rounds != candidate.num_rounds {
        return Err(StageDivergence {
            stage,
            kind: DivergenceKind::NumRounds {
                expected: reference.num_rounds,
                actual: candidate.num_rounds,
            },
        });
    }

    if reference.poly_degree != candidate.poly_degree {
        return Err(StageDivergence {
            stage,
            kind: DivergenceKind::PolyDegree {
                expected: reference.poly_degree,
                actual: candidate.poly_degree,
            },
        });
    }

    if reference.round_poly_coeffs.len() != candidate.round_poly_coeffs.len() {
        return Err(StageDivergence {
            stage,
            kind: DivergenceKind::RoundCount {
                expected: reference.round_poly_coeffs.len(),
                actual: candidate.round_poly_coeffs.len(),
            },
        });
    }

    for (round, (ref_coeffs, cand_coeffs)) in reference
        .round_poly_coeffs
        .iter()
        .zip(&candidate.round_poly_coeffs)
        .enumerate()
    {
        if ref_coeffs.len() != cand_coeffs.len() {
            return Err(StageDivergence {
                stage,
                kind: DivergenceKind::CoeffCount {
                    round,
                    expected: ref_coeffs.len(),
                    actual: cand_coeffs.len(),
                },
            });
        }
        for (coeff, (a, b)) in ref_coeffs.iter().zip(cand_coeffs).enumerate() {
            if a != b {
                return Err(StageDivergence {
                    stage,
                    kind: DivergenceKind::Coefficient {
                        round,
                        coeff,
                        expected: format!("{a:?}"),
                        actual: format!("{b:?}"),
                    },
                });
            }
        }
    }

    if reference.evals.len() != candidate.evals.len() {
        return Err(StageDivergence {
            stage,
            kind: DivergenceKind::EvalCount {
                expected: reference.evals.len(),
                actual: candidate.evals.len(),
            },
        });
    }

    for (index, (a, b)) in reference.evals.iter().zip(&candidate.evals).enumerate() {
        if a != b {
            return Err(StageDivergence {
                stage,
                kind: DivergenceKind::Eval {
                    index,
                    expected: format!("{a:?}"),
                    actual: format!("{b:?}"),
                },
            });
        }
    }

    Ok(())
}
