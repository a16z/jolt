use super::super::super::types::ProtocolArtifactConfig;

pub(super) fn generated_standalone_manifest(
    config: &ProtocolArtifactConfig,
    crate_name: &str,
    dependencies: Vec<String>,
    dependency_root: &str,
) -> String {
    let patch_section = if config.crates_io_patches.is_empty() {
        String::new()
    } else {
        format!(
            "\n[patch.crates-io]\n{}\n",
            config.crates_io_patches.join("\n")
        )
    };
    let dependencies = dependencies
        .into_iter()
        .map(|name| standalone_dependency_entry(config, dependency_root, &name))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "[package]\nname = \"{crate_name}\"\nversion = \"0.0.0\"\nedition = \"2021\"\n{patch_section}\n[dependencies]\n{dependencies}\n"
    )
}

fn standalone_dependency_entry(
    config: &ProtocolArtifactConfig,
    dependency_root: &str,
    package: &str,
) -> String {
    config
        .standalone_dependency_overrides
        .iter()
        .find(|dependency| dependency.package == package)
        .map(|dependency| dependency.manifest_entry.clone())
        .unwrap_or_else(|| format!("{package} = {{ path = \"{dependency_root}/{package}\" }}"))
}
