//! Dory polynomial commitment scheme for the Jolt zkVM.
//!
//! This crate wraps the external `dory-pcs` crate behind the commitment scheme
//! traits from `jolt-openings`, providing:
//!
//! - [`CommitmentScheme`](jolt_openings::CommitmentScheme) -- commit, prove, verify
//! - [`HomomorphicCommitmentScheme`](jolt_openings::HomomorphicCommitmentScheme) -- batch proofs via RLC
//! - [`StreamingCommitmentScheme`](jolt_openings::StreamingCommitmentScheme) -- chunked commitment
//!
//! # Instance-local parameters
//!
//! Unlike the old `dory_globals.rs` approach which used `static mut` global state,
//! this crate stores all Dory matrix shape parameters in [`DoryParams`], carried
//! on the [`DoryScheme`] instance. This allows multiple independent Dory instances
//! within a single process.
//!
//! # Example
//!
//! ```ignore
//! use jolt_dory::{DoryScheme, DoryParams};
//! use jolt_openings::CommitmentScheme;
//!
//! let params = DoryParams::from_dimensions(4, 4);
//! let scheme = DoryScheme::new(params);
//! let setup = DoryScheme::setup_prover(4);
//! ```

pub mod error;
pub mod params;
pub mod scheme;
pub mod streaming;
pub mod types;

mod transcript;

pub use params::DoryParams;
pub use scheme::DoryScheme;
pub use streaming::DoryStreamingCommitter;
pub use types::{
    DoryBatchedProof, DoryCommitment, DoryHint, DoryPartialCommitment, DoryProof, DoryProverSetup,
    DoryVerifierSetup,
};
