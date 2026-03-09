//! Lightweight Jolt proof verification.
//!
//! This crate defines the canonical proof types, verification key, verifier
//! stage trait, and top-level [`verify`] function. External consumers import
//! this crate to verify Jolt proofs — no prover dependencies, no rayon,
//! no compute backend.
//!
//! # Crate structure
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`proof`] | `JoltProof`, `SumcheckStageProof`, `BatchOpeningProofs` |
//! | [`key`] | `JoltVerifyingKey` |
//! | [`error`] | `JoltError` |
//! | [`stage`] | `VerifierStage` trait |
//! | [`verifier`] | `verify()` — top-level verification pipeline |
//!
//! # Usage
//!
//! ```ignore
//! use jolt_verifier::{verify, JoltProof, JoltVerifyingKey};
//!
//! let vk: JoltVerifyingKey<Fr, DoryScheme> = /* ... */;
//! let proof: JoltProof<Fr, DoryScheme> = /* ... */;
//! let mut stages: Vec<Box<dyn VerifierStage<_, _, _>>> = /* ... */;
//! let mut transcript = Blake2bTranscript::new(b"jolt");
//! let (r_x, r_y) = verify(&proof, &vk, &mut stages, &mut transcript, |c| c.into())?;
//! ```

pub mod error;
pub mod key;
pub mod proof;
pub mod stage;
pub mod verifier;

pub use error::JoltError;
pub use key::JoltVerifyingKey;
pub use proof::{BatchOpeningProofs, JoltProof, SumcheckStageProof};
pub use stage::VerifierStage;
pub use verifier::{verify, verify_openings, verify_spartan};
