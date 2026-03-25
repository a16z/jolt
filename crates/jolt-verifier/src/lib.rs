//! Jolt proof verification.
//!
//! This crate verifies Jolt proofs by walking the protocol graph from
//! [`jolt_ir`]. No prover dependencies, no rayon, no compute backend.

pub mod config;
pub mod error;
pub mod key;
pub mod proof;
pub mod verifier;
pub mod verify;

pub use config::{OneHotConfig, OneHotParams, ProverConfig, ReadWriteConfig};
pub use error::JoltError;
pub use key::JoltVerifyingKey;
pub use proof::{JoltProof, StageProof};
pub use verifier::{verify_openings, verify_spartan};
pub use verify::{build_symbol_table, gamma_powers, verify_from_graph};
