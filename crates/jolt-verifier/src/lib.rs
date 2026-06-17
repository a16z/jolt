//! Verifier model crate for Jolt proofs.

pub mod compat;
pub mod config;
pub mod error;
pub mod preprocessing;
pub mod proof;
pub mod stages;
pub mod verifier;

pub use config::{
    validate_proof_config, validate_protocol_config, AdviceLatticeConfig, FieldInlineLatticeConfig,
    IncrementCommitmentMode, JoltProtocolConfig, LatticeConfig, PackedWitnessConfig, PcsFamily,
    PcsFamilyFlags, ProgramMode, ZkConfig, JOLT_VERIFIER_CONFIG, SELECTED_FIELD_INLINE_CONFIG,
};
pub use error::VerifierError;
pub use preprocessing::{
    CommittedProgramPreprocessing, JoltVerifierPreprocessing, ProgramPreprocessing,
};
pub use proof::{
    validate_commitment_payload_family, AkitaCommitmentPayload, ClearProofClaims,
    CommitmentPayload, DoryCommitmentPayload, JoltProof, JoltProofClaims,
};
#[cfg(feature = "field-inline")]
pub use proof::{FieldInlineCommitments, FieldRegistersCommitments};
pub use verifier::{stage8_batch_statement, verify, CheckedInputs};
