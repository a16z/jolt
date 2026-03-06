//! Generic sumcheck protocol engine for interactive and non-interactive proofs.
//!
//! This crate provides a clean implementation of the sumcheck
//! protocol, the workhorse of modern SNARK constructions. It is
//! backend-agnostic: any field, transcript, and witness representation
//! can be plugged in.
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
//! | [`prover`] | [`SumcheckProver`] engine + [`SumcheckWitness`] trait |
//! | [`verifier`] | [`SumcheckVerifier`] engine |
//! | [`batched`] | Batched prover/verifier via random linear combination |
//! | [`streaming`] | [`StreamingSumcheckProver`] trait for memory-constrained provers |
//! | [`handler`] | [`RoundHandler`] / [`RoundVerifier`] — strategy traits for clear vs. committed mode |
//! | [`error`] | [`SumcheckError`] variants |

pub mod batched;
pub mod claim;
pub mod error;
pub mod handler;
pub mod proof;
pub mod prover;
pub mod streaming;
pub mod verifier;

pub use batched::{BatchedSumcheckProver, BatchedSumcheckVerifier};
pub use claim::SumcheckClaim;
pub use error::SumcheckError;
pub use handler::{ClearRoundHandler, ClearRoundVerifier, RoundHandler, RoundVerifier};
pub use proof::SumcheckProof;
pub use prover::{SumcheckProver, SumcheckWitness};
pub use streaming::StreamingSumcheckProver;
pub use verifier::SumcheckVerifier;

#[cfg(test)]
mod tests;
