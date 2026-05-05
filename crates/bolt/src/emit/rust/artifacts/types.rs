mod config;
mod extensions;
mod generated;
mod references;
mod stage;

pub use config::{ProtocolArtifactConfig, ProtocolStandaloneDependency};
pub use extensions::{
    ProtocolArtifactExtension, ProtocolProverApiExtension, ProtocolVerifierApiExtension,
};
pub use generated::{
    ArtifactCrateRole, GeneratedCrate, GeneratedFile, ProtocolRuntimeModule, ProtocolRustArtifact,
};
pub use references::{ProtocolCrateRef, RustTypeRef};
pub use stage::{ProtocolStage, ProtocolStageKind};
