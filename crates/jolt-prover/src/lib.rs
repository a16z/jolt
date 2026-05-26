//! Protocol orchestration for modular Jolt proving.
//!
//! This crate owns proof order, transcript scheduling, and typed verifier-proof
//! construction. Concrete compute implementations live in `jolt-backends`.

mod config;
mod error;
mod preprocessing;
mod prover;
pub mod stages;

pub use config::{ProverConfig, ProverFeatureSet};
pub use error::ProverError;
pub use preprocessing::JoltProverPreprocessing;
pub use prover::prove;
