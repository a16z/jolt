//! Verifier model crate for Jolt proofs.

pub mod compat;
pub mod config;
pub mod error;
pub mod pcs_assist;
pub mod preprocessing;
pub mod proof;
pub mod stages;
pub mod verifier;

pub use config::{
    validate_proof_config, zk_vector_commitment_capacity_requirement, JoltProtocolConfig,
    JoltProtocolConfigSummary, ZkConfig, JOLT_VERIFIER_CONFIG, SELECTED_FIELD_INLINE_CONFIG,
};
pub use error::VerifierError;
pub use pcs_assist::{
    NoPcsAssist, NoPcsAssistConfig, NoPcsAssistProof, PcsAssistClearInput, PcsAssistZkInput,
    PcsProofAssist,
};
pub use preprocessing::JoltVerifierPreprocessing;
pub use proof::{ClearProofClaims, JoltProof, JoltProofClaims};
#[cfg(feature = "field-inline")]
pub use proof::{FieldInlineCommitments, FieldRegistersCommitments};
pub use verifier::{
    absorb_transcript_commitments, absorb_transcript_preamble, validate_inputs_from_parts, verify,
    verify_until_stage1, CheckedInputs, PreStage1VerifierState, ProofTranscriptConfig,
};
