#[rustfmt::skip]
pub mod prover;
pub mod stages;

pub use prover::{
    prove_jolt, DefaultJoltTranscript, JoltProveError, JoltProverArtifacts, JoltProverInputs,
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
];

pub fn generated_stage_names() -> impl Iterator<Item = &'static str> {
    GENERATED_STAGES.iter().map(|stage| stage.name)
}
