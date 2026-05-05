use super::super::super::types::ProtocolArtifactConfig;

pub(in crate::emit::rust::artifacts::role_api) fn find_kernel_module(
    config: &ProtocolArtifactConfig,
    source: &str,
) -> Option<String> {
    let kernel_import = config.kernel_crate.as_ref()?.import.as_str();
    let prefix = format!("use {kernel_import}::");
    source.lines().find_map(|line| {
        let rest = line.trim_start().strip_prefix(&prefix)?;
        rest.split(|character: char| matches!(character, ':' | '{') || character.is_whitespace())
            .next()
            .filter(|name| !name.is_empty())
            .map(ToOwned::to_owned)
    })
}
