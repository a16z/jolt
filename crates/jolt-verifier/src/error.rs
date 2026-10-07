//! Verifier error types.

#[cfg(feature = "akita-byte-link")]
use jolt_claims::protocols::jolt::lattice::byte_link::{ByteLinkBatch, HistogramGroup};
use jolt_claims::protocols::jolt::{
    JoltChallengeId, JoltCommittedPolynomial, JoltDerivedId, JoltOpeningId, JoltRelationId,
};

use crate::config::JoltProtocolConfig;

#[derive(Debug, thiserror::Error)]
pub enum VerifierError {
    #[error("proof protocol config {got:?} does not match verifier config {expected:?}")]
    ProtocolConfigMismatch {
        expected: JoltProtocolConfig,
        got: JoltProtocolConfig,
    },

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

    #[error("invalid RAM domain size {got}; expected a power of two in [{min}, {max}]")]
    InvalidRamK { got: usize, min: usize, max: usize },

    #[error("invalid verifier memory layout: {reason}")]
    InvalidMemoryLayout { reason: String },

    #[error("invalid precommitted claim-reduction schedule: {reason}")]
    InvalidPrecommittedSchedule { reason: String },

    #[error("invalid committed program preprocessing: {reason}")]
    InvalidCommittedProgram { reason: String },

    #[error("missing stage claim challenge input {id:?}")]
    MissingStageClaimChallenge { id: JoltChallengeId },

    #[error(transparent)]
    ChallengeDraw(#[from] jolt_claims::ChallengeDrawError),

    #[error("missing stage claim public input {id:?}")]
    MissingStageClaimDerived { id: JoltDerivedId },

    #[error("stage {stage} opening inputs {left:?} and {right:?} must have the same evaluation")]
    StageClaimOpeningMismatch {
        stage: String,
        left: JoltOpeningId,
        right: JoltOpeningId,
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

    #[error("final opening proof verification failed: {reason}")]
    FinalOpeningVerificationFailed { reason: String },

    #[error("BlindFold protocol construction failed: {reason}")]
    BlindFoldConstructionFailed { reason: String },

    #[error("BlindFold proof verification failed: {reason}")]
    BlindFoldVerificationFailed { reason: String },

    /// The byte trace opens only through the byte link, which consumes the
    /// routed stage-6b claims; neither front implements the link yet.
    #[cfg(feature = "akita-byte-link")]
    #[error("the byte link is not wired: stage 8 cannot open the byte trace")]
    ByteLinkNotWired,

    #[cfg(feature = "akita-byte-link")]
    #[error("byte link rejected: {0}")]
    ByteLink(#[from] ByteLinkError),
}

/// A byte-link relation the proof fails.
#[cfg(feature = "akita-byte-link")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ByteLinkError {
    #[error("{what} have the wrong length")]
    Shape { what: &'static str },
    #[error("a root denominator of pack {pack} is zero")]
    ZeroRoot { pack: usize },
    #[error("pack {pack}'s trace and table roots are different fractions")]
    Roots { pack: usize },
    #[error("{batch:?} layer {layer}: the children's gate disagrees with the sumcheck")]
    Gate { batch: ByteLinkBatch, layer: usize },
    #[error("pack {pack}'s trace leaf numerator is not eq(r, z)")]
    TraceLeaf { pack: usize },
    #[error("pack {pack}'s table leaf denominator disagrees with its public table")]
    TableLeaf { pack: usize },
    #[error("{group:?} histogram query reduction does not hold at its final point")]
    Query { group: HistogramGroup },
    #[error("the Q reduction does not hold at its final point")]
    Source,
}
