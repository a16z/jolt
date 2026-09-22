//! Verifier error types.

use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltRelationId};
use jolt_riscv::JoltInstructionKind;

use crate::config::JoltProtocolConfig;
use crate::stages::ids::{VerifierChallengeId, VerifierDerivedId, VerifierOpeningId};

#[derive(Debug, thiserror::Error)]
pub enum VerifierError {
    #[error("proof protocol config {got:?} does not match verifier config {expected:?}")]
    ProtocolConfigMismatch {
        expected: JoltProtocolConfig,
        got: JoltProtocolConfig,
    },

    #[error("the protocol axis `{axis}` is not supported on this path: {pending}")]
    ProtocolAxisUnimplemented {
        axis: &'static str,
        pending: &'static str,
    },

    #[error("proof payload {field} is required by the configured protocol but missing")]
    MissingProofPayload { field: &'static str },

    #[error(
        "verifier preprocessing payload {field} is required by the configured protocol but \
         missing"
    )]
    MissingPreprocessingPayload { field: &'static str },

    #[error("proof field {field} must be clear for non-ZK verification")]
    ExpectedClearProof { field: &'static str },

    #[error("proof field {field} must be committed for ZK verification")]
    ExpectedCommittedProof { field: &'static str },

    #[error("clear proof unexpectedly includes a BlindFold proof")]
    UnexpectedBlindFoldProof,

    #[error("committed proof is missing a BlindFold proof")]
    MissingBlindFoldProof,

    #[error("committed proof unexpectedly includes opening claims")]
    UnexpectedOpeningClaims,

    #[error("missing opening claim scalar {id:?}")]
    MissingOpeningClaim { id: VerifierOpeningId },

    #[error("unexpected opening claim scalar {id:?}")]
    UnexpectedOpeningClaim { id: VerifierOpeningId },

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

    #[error("missing stage claim challenge input {id:?}")]
    MissingStageClaimChallenge { id: VerifierChallengeId },

    #[error(transparent)]
    ChallengeDraw(#[from] jolt_claims::ChallengeDrawError),

    #[error("missing stage claim public input {id:?}")]
    MissingStageClaimDerived { id: VerifierDerivedId },

    #[error("stage {stage} opening inputs {left:?} and {right:?} must have the same evaluation")]
    StageClaimOpeningMismatch {
        stage: String,
        left: VerifierOpeningId,
        right: VerifierOpeningId,
    },

    #[error("stage {stage} sumcheck verification failed: {reason}")]
    StageClaimSumcheckFailed { stage: String, reason: String },

    #[error("stage {stage:?} public claim construction failed: {reason}")]
    StageClaimPublicInputFailed {
        stage: JoltRelationId,
        reason: String,
    },

    #[error("stage {stage} sumcheck output does not match evaluated output claim")]
    StageClaimOutputMismatch { stage: usize },

    #[error("invalid final opening commitment count {got}; expected {expected}")]
    InvalidCommitmentCount { expected: usize, got: usize },

    #[error("missing final opening commitment for {polynomial:?}")]
    MissingFinalOpeningCommitment { polynomial: JoltCommittedPolynomial },

    #[error("final opening batch construction failed: {reason}")]
    FinalOpeningBatchFailed { reason: String },

    #[error(
        "the FR limb evaluations do not recompose the stage-6b reduced FieldRdInc claim: the \
         committed limb columns disagree with the proven increment stream"
    )]
    #[cfg(all(feature = "akita", feature = "field-inline"))]
    FieldIncLimbRecompositionMismatch,

    #[error("final opening proof verification failed: {reason}")]
    FinalOpeningVerificationFailed { reason: String },

    #[error("BlindFold protocol construction failed: {reason}")]
    BlindFoldConstructionFailed { reason: String },

    #[error("BlindFold proof verification failed: {reason}")]
    BlindFoldVerificationFailed { reason: String },

    #[error("bytecode carries {kind:?}, which this verifier build has no constraints for")]
    UnsupportedInstruction { kind: JoltInstructionKind },
    #[error("field-inline bytecode side table rejected: {reason}")]
    InvalidFieldInlineBytecode { reason: String },
}
