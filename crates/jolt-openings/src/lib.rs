//! Commitment scheme traits and opening proof accumulators for the Jolt zkVM.
//!
//! This crate provides abstract interfaces for polynomial commitment schemes
//! (PCS), opening proof accumulation, and batch reduction via random linear
//! combination (RLC).
//!
//! # Trait hierarchy
//!
//! - [`CommitmentScheme`]: base commit/open/verify interface.
//! - [`HomomorphicCommitmentScheme`]: additively homomorphic schemes enabling
//!   batch proofs via RLC.
//! - [`StreamingCommitmentScheme`]: incremental/chunked commitment.
//!
//! # Accumulators
//!
//! [`ProverOpeningAccumulator`] and [`VerifierOpeningAccumulator`] collect
//! opening claims during a multi-round protocol and batch them at the end.

mod accumulator;
mod error;
#[cfg(any(test, feature = "test-utils"))]
pub mod mock;
mod reduction;
mod traits;

pub use accumulator::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
pub use error::OpeningsError;
pub use reduction::{rlc_combine, rlc_combine_scalars};
pub use traits::{CommitmentScheme, HomomorphicCommitmentScheme, StreamingCommitmentScheme};
