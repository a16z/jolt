//! Dory-assist verifier errors.

use std::fmt::{Display, Formatter};

use jolt_claims::protocols::dory_assist::{
    DoryAssistChallengeId, DoryAssistOpeningId, DoryAssistPublicId, DoryAssistRelationId,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoryAssistStage {
    CheckedInputs,
    Stage1,
    Stage2,
    Stage3,
    HyraxOpening,
    NativeOutput,
}

impl Display for DoryAssistStage {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::CheckedInputs => "checked inputs",
            Self::Stage1 => "stage 1",
            Self::Stage2 => "stage 2",
            Self::Stage3 => "stage 3",
            Self::HyraxOpening => "Hyrax opening",
            Self::NativeOutput => "native output",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DoryAssistVerifierError {
    #[error("invalid Dory-assist verifier mode: expected {expected}, got {got}")]
    InvalidMode {
        expected: &'static str,
        got: &'static str,
    },

    #[error("invalid Dory-assist proof shape in {component}: {reason}")]
    InvalidProofShape {
        component: &'static str,
        reason: String,
    },

    #[error("Dory-assist checked input mismatch: {reason}")]
    CheckedInputMismatch { reason: String },

    #[error("Dory-assist {stage} claim mismatch: {reason}")]
    StageClaimMismatch {
        stage: DoryAssistStage,
        reason: String,
    },

    #[error("missing Dory-assist opening claim for {id:?}")]
    MissingOpeningClaim { id: DoryAssistOpeningId },

    #[error("missing Dory-assist stage challenge for {id:?}")]
    MissingStageClaimChallenge { id: DoryAssistChallengeId },

    #[error("missing Dory-assist public claim for {id:?}")]
    MissingStageClaimPublic { id: DoryAssistPublicId },

    #[error("Dory-assist {stage} sumcheck failed for {relation:?}: {reason}")]
    StageSumcheckFailed {
        stage: DoryAssistStage,
        relation: DoryAssistRelationId,
        reason: String,
    },

    #[error("Dory-assist {stage} output mismatch: {reason}")]
    StageOutputMismatch {
        stage: DoryAssistStage,
        reason: String,
    },

    #[error("Dory-assist opening claim mismatch: {reason}")]
    OpeningClaimMismatch { reason: String },

    #[error("Dory-assist Hyrax opening verification failed: {0}")]
    HyraxOpeningFailed(#[from] jolt_hyrax::HyraxError),

    #[error("Dory-assist public output mismatch: {reason}")]
    PublicOutputMismatch { reason: String },

    #[error("Dory-assist transcript mismatch: {reason}")]
    TranscriptMismatch { reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_display_names_are_stable() {
        assert_eq!(DoryAssistStage::CheckedInputs.to_string(), "checked inputs");
        assert_eq!(DoryAssistStage::Stage1.to_string(), "stage 1");
        assert_eq!(DoryAssistStage::Stage2.to_string(), "stage 2");
        assert_eq!(DoryAssistStage::Stage3.to_string(), "stage 3");
        assert_eq!(DoryAssistStage::HyraxOpening.to_string(), "Hyrax opening");
        assert_eq!(DoryAssistStage::NativeOutput.to_string(), "native output");
    }

    #[test]
    fn hyrax_error_conversion_preserves_source() {
        let error = DoryAssistVerifierError::from(jolt_hyrax::HyraxError::EvaluationMismatch);

        assert_eq!(
            error,
            DoryAssistVerifierError::HyraxOpeningFailed(jolt_hyrax::HyraxError::EvaluationMismatch)
        );
    }
}
