//! Jolt proof verification and shared protocol types.
//!
//! This crate contains everything needed to verify Jolt proofs — no prover
//! dependencies, no rayon, no compute backend. It also owns the protocol
//! types and input claim formulas shared by prover and verifier.
//!
//! # Crate structure
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`protocol`] | Stage evaluation types, input claim formulas |
//! | [`verify`] | `verify_proof()` — stateless proof verification |
//! | [`proof`] | `JoltProof`, `StageProof` |
//! | [`config`] | `ProverConfig`, `OneHotConfig`, `ReadWriteConfig` |
//! | [`key`] | `JoltVerifyingKey` |
//! | [`error`] | `JoltError` |

pub mod config;
pub mod error;
pub mod key;
pub mod proof;
pub mod protocol;
pub mod verify;

// Legacy descriptor-driven verifier (used by existing tests, will be removed)
#[doc(hidden)]
pub mod verifier;

pub use config::{OneHotConfig, OneHotParams, ProverConfig, ReadWriteConfig};
pub use error::JoltError;
pub use key::JoltVerifyingKey;
pub use proof::{JoltProof, StageProof};
pub use protocol::*;
pub use verify::verify_proof;
pub use verifier::{verify_openings, verify_spartan};
