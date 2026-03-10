//! Lightweight Jolt proof verification.
//!
//! This crate defines the canonical proof types, verification key, configuration
//! types, and top-level [`verify`] function. External consumers import this
//! crate to verify Jolt proofs — no prover dependencies, no rayon, no compute
//! backend.
//!
//! # Crate structure
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`config`] | `ProverConfig`, `OneHotConfig`, `ReadWriteConfig`, `OneHotParams` |
//! | [`proof`] | `JoltProof`, `SumcheckStageProof` |
//! | [`key`] | `JoltVerifyingKey` |
//! | [`error`] | `JoltError` |
//! | [`stage`] | `StageDescriptor` — config-driven stage verification |
//! | [`verifier`] | `verify()` — top-level verification pipeline |

pub mod config;
pub mod error;
pub mod key;
pub mod proof;
pub mod stage;
pub mod verifier;

pub use config::{OneHotConfig, OneHotParams, ProverConfig, ReadWriteConfig};
pub use error::JoltError;
pub use key::JoltVerifyingKey;
pub use proof::{JoltProof, SumcheckStageProof};
pub use stage::StageDescriptor;
pub use verifier::{verify, verify_openings, verify_spartan};
