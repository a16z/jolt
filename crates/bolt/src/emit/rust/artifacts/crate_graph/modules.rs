use crate::ir::Role;

use super::super::types::{ProtocolArtifactConfig, ProtocolRustArtifact};

pub(super) fn generated_stage_module_source(
    config: &ProtocolArtifactConfig,
    role: &Role,
    artifacts: &[ProtocolRustArtifact],
) -> String {
    let mut stage_module_lines = Vec::new();
    stage_module_lines.extend(
        config
            .runtime_modules(role)
            .iter()
            .map(|module| format!("pub mod {};", module.module_name)),
    );
    stage_module_lines.extend(artifacts.iter().map(|artifact| {
        format!(
            "#[rustfmt::skip]\npub mod {};",
            artifact.stage.module_name()
        )
    }));
    format!("{}\n", stage_module_lines.join("\n"))
}
