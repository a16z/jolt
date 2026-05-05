mod standalone;
mod workspace;

use crate::ir::Role;

use super::super::types::ProtocolArtifactConfig;
use standalone::generated_standalone_manifest;
use workspace::generated_workspace_manifest;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ManifestMode<'a> {
    Standalone { dependency_root: &'a str },
    Workspace,
}

pub(super) fn generated_manifest(
    config: &ProtocolArtifactConfig,
    role: &Role,
    manifest_mode: ManifestMode<'_>,
) -> String {
    let crate_name = config.crate_name(role);
    let dependencies = config.dependencies(role);
    match manifest_mode {
        ManifestMode::Standalone { dependency_root } => {
            generated_standalone_manifest(config, crate_name, dependencies, dependency_root)
        }
        ManifestMode::Workspace => {
            generated_workspace_manifest(config, role, crate_name, dependencies)
        }
    }
}
