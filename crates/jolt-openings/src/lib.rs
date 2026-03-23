//! Polynomial commitment scheme traits and opening reduction for the Jolt zkVM.
//!
//! This crate provides abstract interfaces for polynomial commitment schemes
//! (PCS), stateless claim types, and a reduction framework for batching
//! opening claims via random linear combination (RLC).
//!
//! # Trait hierarchy
//!
//! ```text
//!               Commitment            (jolt-crypto: just Output type)
//!                   |
//!           CommitmentScheme           (+ Field, Proof, commit/open/verify)
//!                   |
//!       AdditivelyHomomorphic          (+ combine)
//!                   |
//!         StreamingCommitment          (+ begin/feed/finish)
//! ```
//!
//! # Claims and reduction
//!
//! [`ProverClaim`] and [`VerifierClaim`] are stateless data types collected by
//! the protocol orchestrator. [`OpeningReduction`] transforms many claims into
//! fewer claims; [`RlcReduction`] is the standard implementation for homomorphic
//! schemes.

mod claims;
mod error;
#[cfg(any(test, feature = "test-utils"))]
pub mod mock;
mod reduction;
mod traits;

pub use claims::{CommittedEval, ProverClaim, VerifierClaim, VirtualEval};
pub use error::OpeningsError;
pub use reduction::{rlc_combine, rlc_combine_scalars, OpeningReduction, RlcReduction};
pub mod transcript;

pub use traits::{
    AdditivelyHomomorphic, CommitmentScheme, StreamingCommitment, VcSetupExtractable,
    ZkOpeningScheme,
};
