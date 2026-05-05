use crate::ir::Role;

use super::super::super::EmitError;
use super::super::types::{GeneratedCrate, ProtocolArtifactConfig, ProtocolRustArtifact};
use super::artifact::validate_rust_artifact_imports;
use super::artifacts_by_role::ArtifactsByRole;
use super::files::generated_crate_files;
use super::manifest::ManifestMode;

pub fn assemble_generated_crates(
    config: &ProtocolArtifactConfig,
    artifacts: Vec<ProtocolRustArtifact>,
    dependency_root: &str,
) -> Result<Vec<GeneratedCrate>, EmitError> {
    assemble_generated_crates_with_manifest(
        config,
        artifacts,
        ManifestMode::Standalone { dependency_root },
    )
}

pub fn assemble_workspace_generated_crates(
    config: &ProtocolArtifactConfig,
    artifacts: Vec<ProtocolRustArtifact>,
) -> Result<Vec<GeneratedCrate>, EmitError> {
    assemble_generated_crates_with_manifest(config, artifacts, ManifestMode::Workspace)
}

fn assemble_generated_crates_with_manifest(
    config: &ProtocolArtifactConfig,
    artifacts: Vec<ProtocolRustArtifact>,
    manifest_mode: ManifestMode<'_>,
) -> Result<Vec<GeneratedCrate>, EmitError> {
    let mut artifacts_by_role = ArtifactsByRole::default();
    for artifact in artifacts {
        validate_rust_artifact_imports(config, &artifact)?;
        artifacts_by_role.push(artifact);
    }
    Ok(artifacts_by_role
        .into_role_artifacts()
        .into_iter()
        .map(|(role, artifacts)| generated_crate(config, role, artifacts, manifest_mode))
        .collect())
}

fn generated_crate(
    config: &ProtocolArtifactConfig,
    role: Role,
    mut artifacts: Vec<ProtocolRustArtifact>,
    manifest_mode: ManifestMode<'_>,
) -> GeneratedCrate {
    artifacts.sort_by_key(|artifact| artifact.stage.order());
    let crate_name = config.crate_name(&role).to_owned();
    let files = generated_crate_files(config, &role, artifacts, manifest_mode);
    GeneratedCrate { crate_name, files }
}
