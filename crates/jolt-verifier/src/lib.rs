//! Verifier model crate for Jolt proofs.

pub mod compat;
pub mod error;
pub mod proof;
pub mod verifier;
pub mod zk;

pub use error::VerifierError;
pub use proof::JoltProof;
pub use zk::ProofMode;
