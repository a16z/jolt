pub mod stages;
#[rustfmt::skip]
pub mod verifier;

pub use verifier::{
    default_verifier_programs, verify_jolt, verify_jolt_evaluation_proof,
    verify_jolt_with_programs, JoltEvaluationProof, JoltEvaluationProofError, JoltNamedEval,
    JoltProof, JoltStageProof, JoltSumcheckOutput, JoltVerificationArtifacts, JoltVerifierInputs,
    JoltVerifierPrograms, JoltVerifyError,
};

pub const TRANSCRIPT_LABEL: &[u8] = b"Jolt";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeneratedStage {
    pub name: &'static str,
    pub module: &'static str,
    pub ordinal: usize,
}

pub const GENERATED_STAGES: &[GeneratedStage] = &[
    GeneratedStage {
        name: "commitment",
        module: "commitment",
        ordinal: 0,
    },
    GeneratedStage {
        name: "stage1_outer",
        module: "stage1_outer",
        ordinal: 1,
    },
    GeneratedStage {
        name: "stage2",
        module: "stage2",
        ordinal: 2,
    },
    GeneratedStage {
        name: "stage3",
        module: "stage3",
        ordinal: 3,
    },
    GeneratedStage {
        name: "stage4",
        module: "stage4",
        ordinal: 4,
    },
    GeneratedStage {
        name: "stage5",
        module: "stage5",
        ordinal: 5,
    },
    GeneratedStage {
        name: "stage6",
        module: "stage6",
        ordinal: 6,
    },
    GeneratedStage {
        name: "stage7",
        module: "stage7",
        ordinal: 7,
    },
    GeneratedStage {
        name: "stage8",
        module: "stage8",
        ordinal: 8,
    },
];

pub fn generated_stage_names() -> impl Iterator<Item = &'static str> {
    GENERATED_STAGES.iter().map(|stage| stage.name)
}
