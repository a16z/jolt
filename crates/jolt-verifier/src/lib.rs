//! Verifier model crate for Jolt proofs.

pub mod compat;
pub mod config;
pub mod error;
pub mod preprocessing;
pub mod proof;
pub mod stages;
pub mod verifier;

pub use config::{
    validate_proof_config, JoltProtocolConfig, ZkConfig, JOLT_VERIFIER_CONFIG,
    SELECTED_FIELD_INLINE_CONFIG,
};
pub use error::VerifierError;
pub use preprocessing::JoltVerifierPreprocessing;
pub use proof::{ClearProofClaims, JoltProof, JoltProofClaims};
#[cfg(feature = "field-inline")]
pub use proof::{FieldInlineCommitments, FieldRegistersCommitments};
pub use verifier::{verify, verify_until_stage1, CheckedInputs, PreStage1VerifierState};
