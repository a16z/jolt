//! Sparse ring challenges for lattice folding.
//!
//! - [`SparseChallenge`]: one sampled sparse polynomial in `F[X]/(X^D + 1)`.
//! - [`SparseChallengeConfig`]: the fixed-weight family `(count_pm1, count_pm2)`
//!   and its norm and entropy policy.
//! - [`OperatorNormRejection`]: the optional operator-norm rejection rule.
//! - [`Challenges`]: sampled folding challenges in claim-major block order.
//! - [`sample_indexed_challenges_from_seed`]: the indexed signed-sparse
//!   sampler, one SHAKE256-keyed stream per coordinate. Akita's fold-response
//!   drawing (`fold_draw.rs`, which stays in Akita) is its consumer.

mod challenges;
mod config;
mod sampler;
mod sparse;

pub use challenges::Challenges;
pub use config::{
    selective_l2_challenge_config, selective_l2_operator_norm_rejection, OperatorNormRejection,
    SparseChallengeConfig, D128_L2_OP_NORM_PM1_COUNT, D128_L2_OP_NORM_PM2_COUNT,
    D128_SELECTIVE_L2_CHALLENGE_CONFIG, D64_L2_OP_NORM_PM1_COUNT, D64_L2_OP_NORM_PM2_COUNT,
    D64_PRODUCTION_PM1_COUNT, D64_PRODUCTION_PM2_COUNT, D64_SELECTIVE_L2_CHALLENGE_CONFIG,
    MIN_FOLD_CHALLENGE_ENTROPY_BITS, PRODUCTION_FOLD_CHALLENGE_RING_DIMS,
};
pub use sampler::{sample_indexed_challenges_from_seed, MAX_STACK_RING_DIM};
pub use sparse::{
    SparseChallenge, SparseChallengeCoefficients, SparseChallengePositions, INLINE_SPARSE_WEIGHT,
};
