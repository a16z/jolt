//! Verifier model crate for Jolt proofs.

pub mod compat;
pub mod error;
pub mod preprocessing;
pub mod proof;
pub mod stages;
pub mod verifier;

pub use error::VerifierError;
pub use preprocessing::JoltVerifierPreprocessing;
pub use proof::{ClearProofClaims, JoltProof, JoltProofClaims};
pub use verifier::{verify, CheckedInputs};
