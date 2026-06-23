//! Protocol orchestration for modular Jolt proving.
//!
//! This crate owns proof order, transcript scheduling, and typed verifier-proof
//! construction. Concrete compute implementations live in `jolt-backends`.

mod api;
#[cfg(feature = "zk")]
mod committed;
mod config;
mod error;
mod preprocessing;
mod prover;
mod stages;
#[cfg(feature = "zk")]
mod zk;

pub use api::BlindFoldProverBackend;
pub use api::{prove, prove_with_components, ProofResult};
pub use config::{ProverConfig, ProverFeatureSet};
pub use error::ProverError;
pub use preprocessing::JoltProverPreprocessing;
