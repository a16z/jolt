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
#[cfg(all(feature = "core-fixtures", feature = "zk"))]
pub use verifier::{audit_zk_blindfold_protocol_shape, ZkBlindFoldProtocolShape};
pub use verifier::{verify, BlindFoldProofVerifier, CheckedInputs};
