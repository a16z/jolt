//! Akita-specific sumcheck stage implementations.
//!
//! Generic sumcheck proof types, traits, and drivers live in `akita-sumcheck`.
//! This module keeps the Akita stage-1/stage-2 instances and the prover-internal
//! two-round-prefix optimization beside the protocol code they depend on.

pub mod akita_stage1;
pub mod akita_stage1_tree;
pub mod akita_stage2;
pub mod setup_sumcheck;
pub mod two_round_prefix;

pub use akita_stage1_tree::AkitaStage1Prover;
pub use akita_stage2::AkitaStage2Prover;
pub use setup_sumcheck::{SetupSumcheckProver, SetupSumcheckProverOutput};

// --- Shared helpers ------------------------------------------------------

use akita_field::FieldCore;

/// Fold a pair of adjacent evaluations in a full-width row at a challenge `r`,
/// with implicit zero-padding when the index falls past the end.
#[inline]
pub(crate) fn fold_full_prefix_pair<E: FieldCore>(row: &[E], left: usize, r: E) -> E {
    let v0 = row.get(left).copied().unwrap_or_else(E::zero);
    let v1 = row.get(left + 1).copied().unwrap_or_else(E::zero);
    v0 + r * (v1 - v0)
}
