//! Protocol-level Fiat-Shamir challenge samplers.
//!
//! Public surface:
//!
//! - [`SparseChallenge`] — the dependency-light data type representing one
//!   sampled sparse polynomial in `F[X]/(X^D + 1)`.
//! - [`SparseChallengeConfig`] — fixed-weight sparse family `(count_pm1, count_pm2)`
//!   exposing policy questions like `l1_norm()` / `infinity_norm()` / `validate()`
//!   to `akita-config`, `akita-types`, and `akita-planner`.
//! - [`FoldDraw`] and [`ForkFoldDraw`] — fold-challenge drawing from a
//!   fold-response fork, shared by both roles.
//! - [`Challenges`] — sampled folding challenges in claim-major block order.
//!
//! Sampling uses the signed-sparse path in a private `sampler` submodule. The
//! SHAKE256-backed XOF cursor is crate-internal and not part of the public API.

mod challenge;
mod challenges;
mod config;
mod fold_draw;
mod sampler;

pub use challenge::{
    SparseChallenge, SparseChallengeCoefficients, SparseChallengePositions, INLINE_SPARSE_WEIGHT,
};
pub use challenges::Challenges;
pub use config::{
    selective_l2_challenge_config, selective_l2_operator_norm_rejection, OperatorNormRejection,
    SparseChallengeConfig, D128_L2_OP_NORM_PM1_COUNT, D128_L2_OP_NORM_PM2_COUNT,
    D128_SELECTIVE_L2_CHALLENGE_CONFIG, D64_L2_OP_NORM_PM1_COUNT, D64_L2_OP_NORM_PM2_COUNT,
    D64_PRODUCTION_PM1_COUNT, D64_PRODUCTION_PM2_COUNT, D64_SELECTIVE_L2_CHALLENGE_CONFIG,
    MIN_FOLD_CHALLENGE_ENTROPY_BITS, PRODUCTION_FOLD_CHALLENGE_RING_DIMS,
};
pub use fold_draw::{
    fold_challenge_sample_label, FoldChallengeDrawDomain, FoldDraw, FoldDrawRecord, ForkFoldDraw,
    FOLD_CHALLENGE_SEED_LEN,
};
