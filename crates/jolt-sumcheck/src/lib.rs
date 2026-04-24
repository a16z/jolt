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
//! | [`batched`] | [`BatchedSumcheckVerifier`] — batched verification via RLC |
//! | [`round`] | [`RoundVerifier`] — strategy trait for clear vs. committed mode |
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
//! ## Round verification strategy
//! - [`RoundVerifier<F>`] — trait controlling how round data is absorbed into the
//!   transcript and checked. Enables both clear and committed (ZK) verification modes.
//! - [`ClearRoundVerifier`] — cleartext implementation: checks
//!   `poly(0) + poly(1) == running_sum` and absorbs coefficients directly.
//!
//! # Dependency position
//!
//! ```text
//! jolt-field      ─┐
//! jolt-poly       ─┼─> jolt-sumcheck
//! jolt-transcript ─┘
//! ```
//!

pub mod batched;
pub mod claim;
pub mod error;
pub mod proof;
pub mod round;
pub mod verifier;

#[cfg(test)]
mod tests;

pub use batched::BatchedSumcheckVerifier;
pub use claim::{EvaluationClaim, SumcheckClaim};
pub use error::SumcheckError;
pub use proof::SumcheckProof;
pub use round::{ClearRoundVerifier, RoundVerifier};
pub use verifier::SumcheckVerifier;
