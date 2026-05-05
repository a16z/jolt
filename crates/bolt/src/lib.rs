pub mod dialects;
pub mod emit;
pub mod ir;
pub mod mlir;
pub mod pass;
pub mod protocols;
pub mod schema;

pub use emit::rust::{
    assemble_generated_crates, assemble_workspace_generated_crates, protocol_rust_artifact,
    validate_rust_artifact_imports, write_generated_crates, ArtifactCrateRole, EmitError,
    GeneratedCrate, GeneratedFile, ProtocolArtifactConfig, ProtocolArtifactExtension,
    ProtocolCrateRef, ProtocolProverApiExtension, ProtocolRuntimeModule, ProtocolRustArtifact,
    ProtocolStage, ProtocolStageKind, ProtocolStandaloneDependency, ProtocolVerifierApiExtension,
    RustSourceFile, RustTypeRef,
};
pub use ir::{
    BoltModule, Compute, Concrete, Cpu, Diagnostic, Party, Phase, Protocol, Role, TextMlir,
};
pub use mlir::{MeliorContext, MlirError};
pub use pass::{
    derive_prover_role, derive_verifier_role, lower_compute_to_cpu, lower_party_to_compute,
    lower_piop_and_fiat_shamir, project_party, project_prover_party, project_verifier_party,
    resolve_compute_kernels_with, verify_concrete_transcript, ComputeKernelSpec, KernelRegistry,
    PartyToComputeLowering, VerifyError,
};
pub use schema::{
    verify_compute_schema, verify_concrete_schema, verify_cpu_schema, verify_party_schema,
    verify_protocol_schema, SchemaError,
};
