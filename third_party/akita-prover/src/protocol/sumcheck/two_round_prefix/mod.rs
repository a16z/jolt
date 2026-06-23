//! Prover-internal two-round-prefix kernels for Akita stages 1 and 2.
//!
//! When the stage-specific prefix gate fires, the first two rounds of each
//! stage's sumcheck can be collapsed into a single bivariate evaluation over
//! a 4-value inner-dimension quad. The prover builds a local compressed grid
//! and immediately reconstructs the two ordinary sumcheck round messages from
//! it. Those reconstructed messages are then passed to the normal generic
//! sumcheck drivers and serialized as ordinary `SumcheckProof` or
//! `EqFactoredSumcheckProof` rounds.
//!
//! The bivariate-skip grids in this module are not part of the public proof
//! object or verifier API. They are transient prover-side payloads used to avoid
//! expensive scans over compact witness tables before the witness is folded to
//! round 2.
//!
//! Point semantics for the evaluation domains:
//!
//! - Finite points are ordinary evaluations of the bilinear multilinear
//!   extension over the quad.
//! - `Infinity` means "take the leading coefficient in that coordinate".
//!
//! Stage 1 (`b = 4`): domain `{0, 1, Infinity}^2`, 9-point internal grid with
//! the four Boolean corners omitted (5 cached values).
//!
//! Stage 1 (`b = 8`): domain `{0, 1, -1, 2, Infinity}^2`, 25-point internal
//! grid with the four Boolean corners omitted (21 cached values).
//!
//! Stage 2 (`b = 8`): domain `{0, 1, Infinity}^2`, 9-point internal grid. The
//! norm and relation families each cache a compressed grid with one Boolean
//! corner omitted (8 values each), recovered via the known claim before ordinary
//! round polynomials are emitted.
//!
//! This directory keeps the transient two-round prefix optimization split by
//! shared lookup machinery and stage-specific state machines.

mod common;
mod stage1;
mod stage2;

#[cfg(all(test, not(feature = "zk")))]
mod tests;

pub(crate) use common::{
    stage1_b4_s_digit_from_compact_s, stage1_b8_s_digit_from_compact_s, stage2_b4_w_digit,
    stage2_b8_w_digit,
};
pub(crate) use stage1::{
    build_stage1_bivariate_skip_proof_from_s_compact, can_use_stage1_two_round_prefix,
    Stage1BivariateSkipState,
};
pub(crate) use stage2::{
    build_stage2_bivariate_skip_proof_from_compact, can_use_stage2_two_round_prefix,
    Stage2BivariateSkipState,
};
