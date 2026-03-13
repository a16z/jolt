//! Dory polynomial commitment scheme for the Jolt zkVM.
//!
//! This crate wraps the external `dory-pcs` crate behind the commitment scheme
//! traits from `jolt-openings`, providing:
//!
//! - [`CommitmentScheme`](jolt_openings::CommitmentScheme) -- commit, open, verify
//! - [`AdditivelyHomomorphic`](jolt_openings::AdditivelyHomomorphic) -- linear combination of commitments
//! - [`StreamingCommitment`](jolt_openings::StreamingCommitment) -- chunked commitment
//! - [`ZkOpeningScheme`](jolt_openings::ZkOpeningScheme) -- zero-knowledge opening proofs
//! - [`VcSetupExtractable`](jolt_openings::VcSetupExtractable) -- extract Pedersen setup from Dory SRS
//!
//! # Instance-local parameters
//!
//! All Dory matrix shape parameters are stored in [`DoryParams`], carried on the
//! [`DoryScheme`] instance, enabling multiple independent Dory instances within a
//! single process.
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
pub mod optimizations;
pub mod params;
pub mod scheme;
pub mod streaming;
pub mod types;

mod transcript;

pub use params::DoryParams;
pub use scheme::DoryScheme;
pub use types::{
    DoryCommitment, DoryHint, DoryPartialCommitment, DoryProof, DoryProverSetup, DoryVerifierSetup,
};
