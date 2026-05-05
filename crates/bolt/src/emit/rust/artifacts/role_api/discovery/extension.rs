use super::super::super::types::{
    ProtocolArtifactConfig, ProtocolArtifactExtension, ProtocolRustArtifact,
};
use super::super::{CommitmentRustApi, StageRustApi};

pub(in crate::emit::rust::artifacts::role_api) fn active_role_api_extension<'a>(
    config: &'a ProtocolArtifactConfig,
    stages: &[StageRustApi],
    commitment: &Option<CommitmentRustApi>,
    artifacts: &[ProtocolRustArtifact],
) -> Option<&'a ProtocolArtifactExtension> {
    let extension = config.role_api_extension.as_ref()?;
    extension
        .is_active_with(
            commitment.is_some(),
            |required| {
                stages
                    .iter()
                    .any(|stage| stage.field_name.as_str() == required)
            },
            |required| {
                artifacts
                    .iter()
                    .any(|artifact| artifact.stage.module_name() == required)
            },
        )
        .then_some(extension)
}
