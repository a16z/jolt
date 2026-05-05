mod artifacts;
mod source;

pub use artifacts::{
    assemble_generated_crates, assemble_workspace_generated_crates, protocol_rust_artifact,
    validate_rust_artifact_imports, write_generated_crates, ArtifactCrateRole, GeneratedCrate,
    GeneratedFile, ProtocolArtifactConfig, ProtocolArtifactExtension, ProtocolCrateRef,
    ProtocolProverApiExtension, ProtocolRuntimeModule, ProtocolRustArtifact, ProtocolStage,
    ProtocolStageKind, ProtocolStandaloneDependency, ProtocolVerifierApiExtension, RustTypeRef,
};
pub use source::{EmitError, RustSourceFile};
