mod crate_graph;
mod role_api;
mod support;
mod types;

pub use crate_graph::{
    assemble_generated_crates, assemble_workspace_generated_crates, protocol_rust_artifact,
    validate_rust_artifact_imports, write_generated_crates,
};
pub use types::{
    ArtifactCrateRole, GeneratedCrate, GeneratedFile, ProtocolArtifactConfig,
    ProtocolArtifactExtension, ProtocolCrateRef, ProtocolProverApiExtension, ProtocolRuntimeModule,
    ProtocolRustArtifact, ProtocolStage, ProtocolStageKind, ProtocolStandaloneDependency,
    ProtocolVerifierApiExtension, RustTypeRef,
};
