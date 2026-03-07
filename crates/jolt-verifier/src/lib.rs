//! Lightweight Jolt proof verification.
//!
//! This crate defines the proof types, verification key, verifier stage
//! trait, and top-level `JoltVerifier`. External consumers import only
//! this crate to verify Jolt proofs — no prover dependencies, no rayon,
//! no compute backend.
//!
//! # Crate structure
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`proof`] | `JoltProof`, `JoltVerifyingKey`, `JoltPublicInput` |
//! | [`config`] | `ProverConfig` (deserialized from proof) |
//! | [`error`] | `JoltError` |
//! | [`stage`] | `VerifierStage` trait |
//! | [`verifier`] | `JoltVerifier` — top-level verification pipeline |
//!
//! # Usage
//!
//! ```ignore
//! use jolt_verifier::{JoltVerifier, JoltProof, JoltVerifyingKey};
//!
//! let vk: JoltVerifyingKey<DoryScheme> = /* ... */;
//! let proof: JoltProof<DoryScheme> = /* ... */;
//! let verifier = JoltVerifier::new(vk);
//! verifier.verify(&proof, &stages, &mut transcript)?;
//! ```

pub mod config;
pub mod error;
pub mod proof;
pub mod stage;
pub mod verifier;

pub use config::ProverConfig;
pub use error::JoltError;
pub use proof::{JoltProof, JoltPublicInput, JoltVerifyingKey};
pub use stage::VerifierStage;
pub use verifier::JoltVerifier;
