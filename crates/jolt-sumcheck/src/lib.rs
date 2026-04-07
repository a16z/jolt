//! Sumcheck protocol: claims, proofs, and verification.
//!
//! This crate provides the core sumcheck protocol types and verification
//! logic.
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
//! | [`claim`] | [`SumcheckClaim`] — the public statement |
//! | [`proof`] | [`SumcheckProof`] — serializable proof |
//! | [`verifier`] | [`SumcheckVerifier`] engine |
//! | [`batched`] | [`BatchedSumcheckVerifier`] — batched verification via RLC |
//! | [`round`] | [`RoundVerifier`] — strategy trait for clear vs. committed mode |
//! | [`error`] | [`SumcheckError`] variants |
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
pub use claim::SumcheckClaim;
pub use error::SumcheckError;
pub use proof::SumcheckProof;
pub use round::{ClearRoundVerifier, RoundVerifier};
pub use verifier::SumcheckVerifier;
