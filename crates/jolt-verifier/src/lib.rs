//! Canonical verifier crate for Jolt proofs.

pub mod config;
pub mod error;
pub mod preprocessing;
pub mod proof;
pub mod stages;
pub mod verifier;

pub use config::{
    validate_proof_config, CommitmentConfig, JoltProtocolConfig, ZkConfig, JOLT_VERIFIER_CONFIG,
};
pub use error::VerifierError;
pub use preprocessing::{
    CommittedProgramPreprocessing, JoltVerifierPreprocessing, ProgramPreprocessing,
};
pub use proof::{ClearProofClaims, JoltProof, JoltProofClaims};
pub use stages::relations::{
    ConcreteSumcheck, GetPoint, GetValue, InputClaims, OpeningClaim, OutputClaims,
};
pub use stages::zk::committed::zk_vector_commitment_capacity_requirement;
// The `stages::relations` re-export above carries both the `InputClaims` /
// `OutputClaims` traits and the same-named derive macros (jolt-claims re-exports
// both), so `crate::{OutputClaims, InputClaims}` reaches `#[derive(OutputClaims)]`
// and `impl OutputClaims` alike via distinct namespaces.
pub use verifier::{
    absorb_transcript_commitments, absorb_transcript_preamble, validate_inputs_from_parts, verify,
    verify_until_stage1, CheckedInputs, PreStage1VerifierState, ProofTranscriptConfig,
};
