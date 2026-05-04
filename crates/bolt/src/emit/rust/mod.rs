mod artifacts;
mod commitment;
mod stage1;
mod stage2;
mod stage3;
mod stage4;
mod stage5;
mod stage6;
mod stage7;
mod stage8;

pub(crate) fn push_format(source: &mut String, args: std::fmt::Arguments<'_>) {
    use std::fmt::Write as _;

    if source.write_fmt(args).is_err() {
        std::process::abort();
    }
}

pub use artifacts::{
    assemble_generated_crates, assemble_jolt_generated_crates,
    assemble_jolt_workspace_generated_crates, assemble_workspace_generated_crates,
    jolt_artifact_config, jolt_rust_artifact, protocol_rust_artifact,
    validate_jolt_rust_artifact_imports, validate_rust_artifact_imports, write_generated_crates,
    write_jolt_generated_crates, ArtifactCrateRole, GeneratedCrate, GeneratedFile,
    JoltArtifactCrate, JoltGeneratedCrate, JoltGeneratedFile, JoltProtocolStage, JoltRustArtifact,
    ProtocolArtifactConfig, ProtocolCrateRef, ProtocolRustArtifact, ProtocolStage,
    ProtocolStageKind, RustTypeRef,
};
pub use commitment::{
    commitment_cpu_program, emit_commitment_rust, CommitmentBatchPlan, CommitmentCpuProgram,
    CommitmentParams, EmitError, OptionalCommitmentPlan, OptionalSkipPolicy, OracleGeneration,
    OraclePlan, RustSourceFile, TranscriptStep,
};
pub use stage1::{
    emit_stage1_rust, stage1_cpu_program, Stage1CpuProgram, Stage1KernelPlan,
    Stage1OpeningBatchPlan, Stage1OpeningClaimPlan, Stage1Params, Stage1SumcheckBatchPlan,
    Stage1SumcheckClaimPlan, Stage1SumcheckDriverPlan, Stage1SumcheckEvalPlan,
};
pub use stage2::{emit_stage2_rust, stage2_cpu_program, Stage2CpuProgram};
pub use stage3::{emit_stage3_rust, stage3_cpu_program, Stage3CpuProgram};
pub use stage4::{emit_stage4_rust, stage4_cpu_program, Stage4CpuProgram};
pub use stage5::{emit_stage5_rust, stage5_cpu_program, Stage5CpuProgram};
pub use stage6::{emit_stage6_rust, stage6_cpu_program, Stage6CpuProgram};
pub use stage7::{emit_stage7_rust, stage7_cpu_program, Stage7CpuProgram};
pub use stage8::{emit_stage8_rust, stage8_cpu_program, Stage8CpuProgram};
