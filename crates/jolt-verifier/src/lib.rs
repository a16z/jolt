//! Verifier model crate for Jolt proofs.

#[cfg(feature = "akita")]
pub mod akita;
#[cfg(feature = "akita")]
mod akita_witness;
pub mod compat;
pub mod config;
pub mod error;
pub mod preprocessing;
pub mod proof;
pub mod stages;
pub mod verifier;

#[cfg(feature = "akita")]
pub use akita::{
    akita_lattice_protocol_config_for_layout, attach_akita_packed_validity_proof,
    commit_akita_packed_witness, commit_akita_packed_witness_with_config,
    derive_akita_fused_increment_stage6_claims, prove_akita_jolt_final_openings,
    prove_akita_jolt_final_openings_with_precommitted, prove_akita_jolt_packed_validity,
    prove_akita_packed_openings, prove_akita_packed_validity, prove_akita_stage8_clear_openings,
    prove_akita_stage8_clear_openings_with_precommitted, prove_and_attach_akita_opening_proofs,
    prove_and_attach_akita_opening_proofs_with_precommitted, verify_akita_clear,
    AkitaClearVectorCommitment, AkitaFusedIncrementBytecodeSourceComponents,
    AkitaFusedIncrementStage6Claims, AkitaFusedIncrementStage6Derivation,
    AkitaFusedIncrementTranslationSources, AkitaJoltProof, AkitaPackedValidityProofArtifacts,
    AkitaPackedWitnessArtifacts, AkitaPrecommittedOpeningInput, AkitaStage8ClearOpeningProofs,
    AkitaVerifierPreprocessing,
};
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
    validate_akita_commitment_payload_config, validate_commitment_payload_config,
    validate_commitment_payload_family, AkitaCommitmentPayload, ClearOnlyCommitment,
    ClearOnlyVectorCommitment, ClearOnlyVectorCommitmentSetup, ClearProofClaims, CommitmentPayload,
    DoryCommitmentPayload, JoltProof, JoltProofClaims,
};
#[cfg(feature = "field-inline")]
pub use proof::{FieldInlineCommitments, FieldRegistersCommitments};
pub use verifier::{
    stage8_batch_statement, stage8_batch_statement_with_config,
    stage8_batch_statement_with_config_and_transcript, verify, verify_clear,
    verify_clear_with_config, verify_with_config, CheckedInputs,
};
