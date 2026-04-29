pub mod dialects;
pub mod emit;
pub mod ir;
pub mod mlir;
pub mod pass;
pub mod protocols;
pub mod schema;

pub use emit::rust::{
    commitment_cpu_program, emit_commitment_rust, CommitmentBatchPlan, CommitmentCpuProgram,
    CommitmentParams, EmitError, OptionalCommitmentPlan, OptionalSkipPolicy, OracleGeneration,
    OraclePlan, RustSourceFile, TranscriptStep,
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
    build_commitment_protocol, lower_commitment_to_compute, lower_compute_to_cpu,
    verify_jolt_concrete_schema, verify_jolt_party_schema, verify_jolt_protocol_schema,
    JoltProtocolParams,
};
pub use schema::{
    verify_compute_schema, verify_concrete_schema, verify_cpu_schema, verify_party_schema,
    verify_protocol_schema, SchemaError,
};
