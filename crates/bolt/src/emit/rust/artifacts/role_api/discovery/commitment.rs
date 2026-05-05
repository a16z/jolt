use super::super::super::support::upper_camel;
use super::super::super::types::ProtocolRustArtifact;
use super::super::role::RoleApiRole;
use super::super::source_scan::{find_public_item, find_type_with_suffix};
use super::super::CommitmentRustApi;
use super::modules::module_alias;
use super::program::discover_program_binding;

pub(in crate::emit::rust::artifacts::role_api) fn commitment_api(
    artifacts: &[ProtocolRustArtifact],
) -> Option<CommitmentRustApi> {
    let artifact = artifacts
        .iter()
        .find(|artifact| artifact.stage.is_commitment())?;
    let source = artifact.source.source.as_str();
    let role = RoleApiRole::from_role(&artifact.role);
    let artifacts_type = find_type_with_suffix(source, "Artifacts")
        .unwrap_or_else(|| format!("{}Artifacts", upper_camel(artifact.stage.module_name())));
    let error_type = find_public_item(source, "pub enum ", "Error")
        .unwrap_or_else(|| format!("{}Error", upper_camel(artifact.stage.module_name())));
    let input_provider_trait = find_public_item(source, "pub trait ", "InputProvider");
    let program_binding =
        discover_program_binding(source, role.commitment_program_type_suffix(), &["prove_"]);
    Some(CommitmentRustApi {
        field_name: artifact.stage.module_name().to_owned(),
        module_alias: module_alias(artifact.stage.module_name()),
        variant_name: upper_camel(artifact.stage.module_name()),
        artifacts_type,
        error_type,
        program_binding,
        input_provider_trait,
    })
}
