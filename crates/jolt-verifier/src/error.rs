//! Verifier error types.

#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::{
    FieldInlineChallengeId, FieldInlineOpeningId, FieldInlinePublicId,
};
use jolt_claims::protocols::jolt::{
    JoltChallengeId, JoltCommittedPolynomial, JoltOpeningId, JoltPublicId, JoltRelationId,
};

use crate::config::{JoltProtocolConfig, PcsFamily};

#[derive(Debug, thiserror::Error)]
pub enum VerifierError {
    #[error("proof protocol config {got:?} does not match verifier config {expected:?}")]
    ProtocolConfigMismatch {
        expected: JoltProtocolConfig,
        got: JoltProtocolConfig,
    },

    #[error("invalid protocol config: {reason}")]
    InvalidProtocolConfig { reason: String },

    #[error(
        "proof commitment payload family {got:?} does not match selected PCS family {expected:?}"
    )]
    CommitmentPayloadFamilyMismatch { expected: PcsFamily, got: PcsFamily },

    #[error("lattice commitment payload layout digest does not match verifier config")]
    LatticePayloadLayoutDigestMismatch { expected: [u8; 32], got: [u8; 32] },

    #[error(
        "lattice commitment payload D_pack {got} does not match verifier config D_pack {expected}"
    )]
    LatticePayloadDimensionMismatch { expected: usize, got: usize },

    #[error("Akita packing witness commitment failed: {reason}")]
    AkitaCommitmentFailed { reason: String },

    #[error("lattice packed validity proof is missing {field}")]
    MissingLatticePackedValidityProof { field: &'static str },

    #[error("non-lattice proof unexpectedly includes lattice packed validity {field}")]
    UnexpectedLatticePackedValidityProof { field: &'static str },

    #[error(
        "lattice packed validity opening claim count {got} does not match derived statement count {expected}"
    )]
    LatticePackedValidityClaimCountMismatch { expected: usize, got: usize },

    #[error("lattice packed validity sumcheck failed: {reason}")]
    LatticePackedValiditySumcheckFailed { reason: String },

    #[error("lattice packed validity sumcheck output does not match packed opening claims")]
    LatticePackedValidityOutputMismatch,

    #[error("lattice packed validity opening proof verification failed: {reason}")]
    LatticePackedValidityOpeningVerificationFailed { reason: String },

    #[error("proof field {field} must be clear for non-ZK verification")]
    ExpectedClearProof { field: &'static str },

    #[error("proof field {field} must be committed for ZK verification")]
    ExpectedCommittedProof { field: &'static str },

    #[error("proof is missing optional stage proof field {field}")]
    MissingStageProof { field: &'static str },

    #[error("proof unexpectedly includes optional stage proof field {field}")]
    UnexpectedStageProof { field: &'static str },

    #[error("clear proof is missing opening claims")]
    MissingOpeningClaims,

    #[error("clear proof unexpectedly includes a BlindFold proof")]
    UnexpectedBlindFoldProof,

    #[error("committed proof is missing a BlindFold proof")]
    MissingBlindFoldProof,

    #[error("committed proof unexpectedly includes opening claims")]
    UnexpectedOpeningClaims,

    #[error("missing opening claim scalar {id:?}")]
    MissingOpeningClaim { id: JoltOpeningId },

    #[cfg(feature = "field-inline")]
    #[error("missing field-inline opening claim scalar {id:?}")]
    MissingFieldInlineOpeningClaim { id: FieldInlineOpeningId },

    #[cfg(feature = "field-inline")]
    #[error("missing field-inline claim challenge input {id:?}")]
    MissingFieldInlineChallenge { id: FieldInlineChallengeId },

    #[cfg(feature = "field-inline")]
    #[error("missing field-inline claim public input {id:?}")]
    MissingFieldInlinePublic { id: FieldInlinePublicId },

    #[error("unexpected opening claim scalar {id:?}")]
    UnexpectedOpeningClaim { id: JoltOpeningId },

    #[error("vector commitment setup is missing from verifier preprocessing")]
    MissingVectorCommitmentSetup,

    #[error("vector commitment setup capacity {got} is too small; expected at least {required}")]
    InvalidVectorCommitmentCapacity { required: usize, got: usize },

    #[error("program I/O memory layout does not match verifier preprocessing")]
    MemoryLayoutMismatch,

    #[error("public input length {got} exceeds configured maximum {max}")]
    InputTooLarge { got: usize, max: usize },

    #[error("public output length {got} exceeds configured maximum {max}")]
    OutputTooLarge { got: usize, max: usize },

    #[error("invalid trace length {got}; expected a power of two no larger than {max}")]
    InvalidTraceLength { got: usize, max: usize },

    #[error("invalid RAM domain size {got}; expected a power of two in [{min}, {max}]")]
    InvalidRamK { got: usize, min: usize, max: usize },

    #[error("invalid verifier memory layout: {reason}")]
    InvalidMemoryLayout { reason: String },

    #[error("invalid precommitted claim-reduction schedule: {reason}")]
    InvalidPrecommittedSchedule { reason: String },

    #[error("invalid committed program preprocessing: {reason}")]
    InvalidCommittedProgram { reason: String },

    #[error("missing stage claim opening input {id:?}")]
    MissingStageClaimOpening { id: JoltOpeningId },

    #[error("missing stage claim challenge input {id:?}")]
    MissingStageClaimChallenge { id: JoltChallengeId },

    #[error("missing stage claim public input {id:?}")]
    MissingStageClaimPublic { id: JoltPublicId },

    #[error("stage {stage:?} opening inputs {left:?} and {right:?} must have the same evaluation")]
    StageClaimOpeningMismatch {
        stage: JoltRelationId,
        left: JoltOpeningId,
        right: JoltOpeningId,
    },

    #[error("stage {stage:?} claim expressions must evaluate to the same value")]
    StageClaimExpressionMismatch { stage: JoltRelationId },

    #[error("stage {stage:?} sumcheck degree {degree} is invalid")]
    InvalidStageSumcheckDegree {
        stage: JoltRelationId,
        degree: usize,
    },

    #[error("stage {stage:?} compressed sumcheck proof requires a Boolean domain")]
    CompressedStageClaimRequiresBooleanDomain { stage: JoltRelationId },

    #[error("stage {stage:?} sumcheck verification failed: {reason}")]
    StageClaimSumcheckFailed {
        stage: JoltRelationId,
        reason: String,
    },

    #[error("stage {stage:?} public claim construction failed: {reason}")]
    StageClaimPublicInputFailed {
        stage: JoltRelationId,
        reason: String,
    },

    #[error("stage {stage:?} sumcheck output does not match evaluated output claim")]
    StageClaimOutputMismatch { stage: JoltRelationId },

    #[error("invalid final opening commitment count {got}; expected {expected}")]
    InvalidCommitmentCount { expected: usize, got: usize },

    #[error("missing final opening commitment for {polynomial:?}")]
    MissingFinalOpeningCommitment { polynomial: JoltCommittedPolynomial },

    #[error("final opening batch construction failed: {reason}")]
    FinalOpeningBatchFailed { reason: String },

    #[error("final opening proof verification failed: {reason}")]
    FinalOpeningVerificationFailed { reason: String },

    #[error("BlindFold protocol construction failed: {reason}")]
    BlindFoldConstructionFailed { reason: String },

    #[error("BlindFold proof verification failed: {reason}")]
    BlindFoldVerificationFailed { reason: String },
}
