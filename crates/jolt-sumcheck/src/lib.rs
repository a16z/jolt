//! Sumcheck protocol: claims, proofs, and verification.
//!
//! Verifier-side types and logic for the sumcheck protocol, used by the Jolt
//! zkVM. This crate is **verifier-only** and **backend-agnostic**: any field and
//! transcript can be plugged in. Proving is handled by `jolt-zkvm`'s runtime,
//! which drives sumcheck rounds via `ComputeBackend` primitives.
//!
//! # Protocol overview
//!
//! The sumcheck protocol reduces the verification of a claim
//! $$\sum_{x \in \{0,1\}^n} g(x) = C$$
//! to a single evaluation query $g(r_1, \ldots, r_n) = v$ via $n$
//! rounds of interaction. In round $i$ the prover sends a univariate
//! polynomial $s_i(X)$ and the verifier checks $s_i(0) + s_i(1)$ against
//! the running sum, then sets $r_i$ and recurses.
//!
//! # Crate structure
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`claim`] | [`SumcheckClaim`] (input statement) and [`EvaluationClaim`] (reduction output) |
//! | [`proof`] | [`SumcheckProof`] — serializable proof |
//! | [`verifier`] | [`SumcheckVerifier`] engine |
//! | [`batched_verifier`] | [`BatchedSumcheckVerifier`] — batched verification via RLC |
//! | [`round_proof`] | [`RoundProof`] — per-round trait and concrete wire-format impls |
//! | [`error`] | [`SumcheckError`] variants |
//!
//! # Public API
//!
//! ## Types
//! - [`SumcheckClaim<F>`] — the public statement: `num_vars`, `degree`, and `claimed_sum`.
//! - [`EvaluationClaim<F>`] — the oracle evaluation claim `g(r) = v` produced by a
//!   successful reduction; the caller MUST discharge it against the polynomial oracle.
//! - [`SumcheckProof<F>`] — a sequence of univariate round polynomials, one per variable.
//! - [`SumcheckError`] — error variants: `RoundCheckFailed`, `DegreeBoundExceeded`,
//!   `WrongNumberOfRounds`, `EmptyClaims`.
//!
//! ## Verifiers
//! - [`SumcheckVerifier`] — single-instance verifier. Replays the Fiat-Shamir
//!   transcript and checks each round.
//! - [`BatchedSumcheckVerifier`] — batched verification via random linear
//!   combination. Supports claims with different `num_vars` and `degree` bounds
//!   via front-loaded padding.
//!
//! ## Per-round proof types
//! - [`RoundProof<F>`] — trait implemented by anything the verifier can step
//!   through one round at a time: degree bound, sum check, transcript absorb,
//!   evaluation at challenge.
//! - [`UnivariatePoly<F>`](jolt_poly::UnivariatePoly) — raw, unlabelled absorb.
//! - [`LabeledRoundPoly`] — borrowed wrapper adding a `LabelWithCount` prefix.
//! - [`CompressedLabeledRoundPoly`] — borrowed wrapper using the compressed
//!   wire format (omits the linear coefficient).
//!
//! # Dependency position
//!
//! ```text
//! jolt-field      ─┐
//! jolt-poly       ─┼─> jolt-sumcheck
//! jolt-transcript ─┘
//! ```
//!

pub mod batched_verifier;
pub mod claim;
pub mod error;
pub mod proof;
pub mod round_proof;
pub mod verifier;

#[cfg(test)]
mod tests;

pub use batched_verifier::BatchedSumcheckVerifier;
pub use claim::{EvaluationClaim, SumcheckClaim};
pub use error::SumcheckError;
pub use proof::SumcheckProof;
pub use round_proof::{CompressedLabeledRoundPoly, LabeledRoundPoly, RoundProof};
pub use verifier::SumcheckVerifier;
