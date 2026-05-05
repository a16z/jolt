use crate::ir::Role;

use super::super::role_api::{generated_lib, generated_role_api_file};
use super::super::types::{GeneratedFile, ProtocolArtifactConfig, ProtocolRustArtifact};
use super::manifest::{generated_manifest, ManifestMode};
use super::modules::generated_stage_module_source;

pub(super) fn generated_crate_files(
    config: &ProtocolArtifactConfig,
    role: &Role,
    artifacts: Vec<ProtocolRustArtifact>,
    manifest_mode: ManifestMode<'_>,
) -> Vec<GeneratedFile> {
    let mut files = vec![
        GeneratedFile {
            path: "Cargo.toml".to_owned(),
            source: generated_manifest(config, role, manifest_mode),
        },
        GeneratedFile {
            path: "src/lib.rs".to_owned(),
            source: generated_lib(config, role, &artifacts),
        },
        generated_role_api_file(config, role, &artifacts),
        GeneratedFile {
            path: "src/stages/mod.rs".to_owned(),
            source: generated_stage_module_source(config, role, &artifacts),
        },
    ];
    files.extend(
        config
            .runtime_modules(role)
            .iter()
            .map(|module| module.file.clone()),
    );
    files.extend(artifacts.into_iter().map(|artifact| GeneratedFile {
        path: format!("src/stages/{}.rs", artifact.stage.module_name()),
        source: artifact.source.source,
    }));
    files
}
