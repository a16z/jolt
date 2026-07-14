//! Canonical verifier crate for Jolt proofs.

pub mod config;
pub mod error;
pub mod preprocessing;
pub mod proof;
pub mod stages;
pub mod verifier;

pub use config::{validate_proof_config, JoltProtocolConfig, ZkConfig, JOLT_VERIFIER_CONFIG};
pub use error::VerifierError;
pub use preprocessing::{
    CommittedProgramPreprocessing, JoltVerifierPreprocessing, ProgramPreprocessing,
};
pub use proof::{ClearProofClaims, JoltProof, JoltProofClaims};
pub use verifier::{validate_and_seed_transcript, verify, CheckedInputs};
