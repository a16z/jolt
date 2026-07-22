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
#[cfg(not(feature = "akita"))]
pub use verifier::absorb_transcript_commitments;
pub use verifier::{
    absorb_committed_program_commitments, absorb_transcript_preamble, validate_and_seed_transcript,
    validate_inputs_from_parts, verify, verify_until_stage1, CheckedInputs, PreStage1VerifierState,
    ProofTranscriptConfig,
};
#[cfg(not(feature = "akita"))]
pub use verifier::{verify_stages, VerifiedStages};
