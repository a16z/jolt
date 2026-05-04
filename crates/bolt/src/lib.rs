pub mod dialects;
pub mod emit;
pub mod ir;
pub mod mlir;
pub mod pass;
pub mod protocols;
pub mod schema;

pub use emit::rust::{
    assemble_generated_crates, assemble_jolt_generated_crates,
    assemble_jolt_workspace_generated_crates, assemble_workspace_generated_crates,
    commitment_cpu_program, emit_commitment_rust, emit_stage1_rust, emit_stage2_rust,
    emit_stage3_rust, emit_stage4_rust, emit_stage5_rust, emit_stage6_rust, emit_stage7_rust,
    emit_stage8_rust, jolt_artifact_config, jolt_rust_artifact, protocol_rust_artifact,
    stage1_cpu_program, stage2_cpu_program, stage3_cpu_program, stage4_cpu_program,
    stage5_cpu_program, stage6_cpu_program, stage7_cpu_program, stage8_cpu_program,
    validate_jolt_rust_artifact_imports, validate_rust_artifact_imports, write_generated_crates,
    write_jolt_generated_crates, ArtifactCrateRole, CommitmentBatchPlan, CommitmentCpuProgram,
    CommitmentParams, EmitError, GeneratedCrate, GeneratedFile, JoltArtifactCrate,
    JoltGeneratedCrate, JoltGeneratedFile, JoltProtocolStage, JoltRustArtifact,
    OptionalCommitmentPlan, OptionalSkipPolicy, OracleGeneration, OraclePlan,
    ProtocolArtifactConfig, ProtocolCrateRef, ProtocolRustArtifact, ProtocolStage,
    ProtocolStageKind, RustSourceFile, RustTypeRef, Stage1CpuProgram, Stage1KernelPlan,
    Stage1OpeningBatchPlan, Stage1OpeningClaimPlan, Stage1Params, Stage1SumcheckBatchPlan,
    Stage1SumcheckClaimPlan, Stage1SumcheckDriverPlan, Stage1SumcheckEvalPlan, Stage2CpuProgram,
    Stage3CpuProgram, Stage4CpuProgram, Stage5CpuProgram, Stage6CpuProgram, Stage7CpuProgram,
    Stage8CpuProgram, TranscriptStep,
};
pub use ir::{
    BoltModule, Compute, Concrete, Cpu, Diagnostic, Party, Phase, Protocol, Role, TextMlir,
};
pub use mlir::{MeliorContext, MlirError};
pub use pass::{
    derive_prover_role, derive_verifier_role, lower_piop_and_fiat_shamir, project_party,
    project_prover_party, project_verifier_party, verify_concrete_transcript, VerifyError,
};
pub use protocols::jolt::{
    build_commitment_protocol, build_stage1_outer_protocol, build_stage2_protocol,
    build_stage3_protocol, build_stage4_protocol, build_stage5_protocol, build_stage6_protocol,
    build_stage7_protocol, build_stage8_protocol, lower_commitment_to_compute,
    lower_compute_to_cpu, lower_stage1_to_compute, lower_stage2_to_compute,
    lower_stage3_to_compute, lower_stage4_to_compute, lower_stage5_to_compute,
    lower_stage6_to_compute, lower_stage7_to_compute, lower_stage8_to_compute,
    resolve_compute_kernels, verify_jolt_concrete_schema, verify_jolt_party_schema,
    verify_jolt_protocol_schema, JoltProtocolParams,
};
pub use schema::{
    verify_compute_schema, verify_concrete_schema, verify_cpu_schema, verify_party_schema,
    verify_protocol_schema, SchemaError,
};
