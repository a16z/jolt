//! Canonical verifier crate for Jolt proofs.

// Lets `jolt-verifier-derive` emit absolute `::jolt_verifier::` paths that
// resolve both here and in downstream crates deriving their own batches (the
// `jolt-claims` convention).
extern crate self as jolt_verifier;

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
pub use verifier::{
    absorb_committed_program_commitments, absorb_transcript_commitments,
    absorb_transcript_preamble, validate_inputs_from_parts, verify, verify_until_stage1,
    CheckedInputs, PreStage1VerifierState, ProofTranscriptConfig,
};
