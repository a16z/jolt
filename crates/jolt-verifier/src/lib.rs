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
pub use stages::relations::{
    GetPoint, GetValue, InputClaims, OpeningClaim, OutputClaims, SumcheckInstance,
};
pub use stages::zk::committed::zk_vector_commitment_capacity_requirement;
// Derive macros share the names of the traits they implement (distinct
// namespaces), so `#[derive(OutputClaims)]` and `impl OutputClaims` can both be
// reached as `crate::{OutputClaims, InputClaims}`.
pub use jolt_verifier_derive::{InputClaims, OutputClaims};
pub use verifier::{
    transcript_instance, validate_inputs_from_parts, verify, verify_until_stage1, CheckedInputs,
    PreStage1VerifierState, ProofTranscriptConfig,
};
