//! Verifier error types.

use jolt_claims::protocols::jolt::{
    JoltChallengeId, JoltCommittedPolynomial, JoltOpeningId, JoltPublicId, JoltStageId,
};

#[derive(Debug, thiserror::Error)]
pub enum VerifierError {
    #[error("proof field {field} must be clear for non-ZK verification")]
    ExpectedClearProof { field: &'static str },

    #[error("proof field {field} must be committed for ZK verification")]
    ExpectedCommittedProof { field: &'static str },

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

    #[error("invalid RAM domain size {got}; expected a power of two")]
    InvalidRamK { got: usize },

    #[error("missing stage claim opening input {id:?}")]
    MissingStageClaimOpening { id: JoltOpeningId },

    #[error("missing stage claim challenge input {id:?}")]
    MissingStageClaimChallenge { id: JoltChallengeId },

    #[error("missing stage claim public input {id:?}")]
    MissingStageClaimPublic { id: JoltPublicId },

    #[error("stage {stage:?} opening inputs {left:?} and {right:?} must have the same evaluation")]
    StageClaimOpeningMismatch {
        stage: JoltStageId,
        left: JoltOpeningId,
        right: JoltOpeningId,
    },

    #[error("stage {stage:?} claim expressions must evaluate to the same value")]
    StageClaimExpressionMismatch { stage: JoltStageId },

    #[error("stage {stage:?} sumcheck degree {degree} is invalid")]
    InvalidStageSumcheckDegree { stage: JoltStageId, degree: usize },

    #[error("stage {stage:?} compressed sumcheck proof requires a Boolean domain")]
    CompressedStageClaimRequiresBooleanDomain { stage: JoltStageId },

    #[error("stage {stage:?} sumcheck verification failed: {reason}")]
    StageClaimSumcheckFailed { stage: JoltStageId, reason: String },

    #[error("stage {stage:?} public claim construction failed: {reason}")]
    StageClaimPublicInputFailed { stage: JoltStageId, reason: String },

    #[error("stage {stage:?} sumcheck output does not match evaluated output claim")]
    StageClaimOutputMismatch { stage: JoltStageId },

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

    #[error("verifier functionality has not been implemented yet")]
    Unimplemented,
}
