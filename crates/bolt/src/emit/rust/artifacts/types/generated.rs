use crate::ir::Role;

use super::super::super::RustSourceFile;
use super::stage::ProtocolStage;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactCrateRole {
    Prover,
    Verifier,
}

impl ArtifactCrateRole {
    pub fn for_role(role: &Role) -> Self {
        match role {
            Role::Prover => Self::Prover,
            Role::Verifier => Self::Verifier,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolRustArtifact {
    pub role: Role,
    pub stage: ProtocolStage,
    pub crate_name: String,
    pub path: String,
    pub source: RustSourceFile,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratedCrate {
    pub crate_name: String,
    pub files: Vec<GeneratedFile>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratedFile {
    pub path: String,
    pub source: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolRuntimeModule {
    pub module_name: String,
    pub file: GeneratedFile,
}
