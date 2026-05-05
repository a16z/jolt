use crate::ir::Role;

use super::super::super::types::ProtocolArtifactConfig;

pub(super) fn generated_workspace_manifest(
    config: &ProtocolArtifactConfig,
    role: &Role,
    crate_name: &str,
    dependencies: Vec<String>,
) -> String {
    let dependencies = dependencies
        .into_iter()
        .map(|name| format!("{name}.workspace = true"))
        .collect::<Vec<_>>()
        .join("\n");
    let role_name = role.as_str();
    let repository = config
        .repository
        .as_ref()
        .map(|repository| format!("repository = \"{repository}\"\n"))
        .unwrap_or_default();
    format!(
        "[package]\nname = \"{crate_name}\"\nversion = \"0.0.0\"\nedition = \"2021\"\nlicense = \"MIT OR Apache-2.0\"\ndescription = \"Bolt-generated {} {role_name} role crate\"\n{repository}\n[lints]\nworkspace = true\n\n[dependencies]\n{dependencies}\n",
        config.protocol_name
    )
}
