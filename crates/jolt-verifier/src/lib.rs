pub mod stages;
#[rustfmt::skip]
pub mod verifier;

pub use stages::{
    stage1_outer::{verify_stage1_outer_with_program, Stage1VerifierProgramPlan},
    stage2::{verify_stage2_with_program, Stage2VerifierProgramPlan},
    stage3::{verify_stage3_with_program, Stage3VerifierProgramPlan},
    stage4::{verify_stage4_with_program, Stage4VerifierProgramPlan},
    stage5::{verify_stage5_with_program, Stage5VerifierProgramPlan},
    stage6::{verify_stage6_with_program, Stage6VerifierProgramPlan},
    stage7::{verify_stage7_with_program, Stage7VerifierProgramPlan},
};

pub use verifier::{
    default_verifier_programs, verify_jolt, verify_jolt_evaluation_proof, verify_jolt_prefix,
    verify_jolt_prefix_with_programs, verify_jolt_through_stage5,
    verify_jolt_through_stage5_with_programs, verify_jolt_through_stage6,
    verify_jolt_through_stage6_with_programs, verify_jolt_through_stage7,
    verify_jolt_through_stage7_with_programs, verify_jolt_with_programs, JoltEvaluationPolicy,
    JoltEvaluationProof, JoltEvaluationProofError, JoltNamedEval, JoltProof, JoltProofSlot,
    JoltStage2RamAccess, JoltStage2RamData, JoltStage2RamOutputLayout, JoltStage6BytecodeEntry,
    JoltStage6BytecodeReadRafData, JoltStage6VerifierData, JoltStageChallengeVector,
    JoltStageExecutionArtifacts, JoltStageOpeningInputValue, JoltStageProof, JoltSumcheckOutput,
    JoltVerificationArtifacts, JoltVerifierCheckpoint, JoltVerifierInputs,
    JoltVerifierProgramError, JoltVerifierProgramPlan, JoltVerifierPrograms, JoltVerifierStepPlan,
    JoltVerifierStepKind, JoltVerifierTarget, JoltVerifierTargetPlan, JoltVerifyError,
    JOLT_TARGET_FULL, JOLT_TARGET_THROUGH_STAGE5, JOLT_TARGET_THROUGH_STAGE6,
    JOLT_TARGET_THROUGH_STAGE7, JOLT_VERIFIER_STEPS, JOLT_VERIFIER_TARGETS, VERIFIER_PROGRAM,
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
