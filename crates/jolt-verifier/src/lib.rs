//! Jolt proof verification.
//!
//! No prover dependencies, no rayon, no compute backend.

pub mod config;
pub mod error;
pub mod key;
pub mod proof;
pub mod verifier;

// verify.rs has the graph-driven verifier skeleton but references old
// Spartan types. R1CS verification is now handled by jolt-r1cs + compiler.
// pub mod verify;

pub use config::{OneHotConfig, OneHotParams, ProverConfig, ReadWriteConfig};
pub use error::JoltError;
pub use key::JoltVerifyingKey;
pub use proof::{JoltProof, StageProof};
pub use verifier::verify_openings;
