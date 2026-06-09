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
    validate_proof_config, JoltProtocolConfig, ZkConfig, JOLT_VERIFIER_CONFIG,
    SELECTED_FIELD_INLINE_CONFIG,
};
pub use error::VerifierError;
pub use pcs_assist::{
    NoPcsAssist, NoPcsAssistConfig, NoPcsAssistProof, PcsAssistClearInput, PcsAssistZkInput,
    PcsProofAssist,
};
pub use preprocessing::{
    CommittedProgramPreprocessing, JoltVerifierPreprocessing, ProgramPreprocessing,
};
pub use proof::{ClearProofClaims, JoltProof, JoltProofClaims};
#[cfg(feature = "field-inline")]
pub use proof::{FieldInlineCommitments, FieldRegistersCommitments};
pub use verifier::{verify, CheckedInputs};
