mod error;
mod types;

use super::super::super::support::upper_camel;
use super::super::super::types::{ProtocolArtifactConfig, ProtocolRustArtifact};
use super::super::role::RoleApiRole;
use super::super::source_scan::find_kernel_module;
use super::super::StageRustApi;
use super::modules::module_alias;
use super::program::discover_program_binding;
use error::discover_stage_error_type;
use types::discover_stage_types;

pub(in crate::emit::rust::artifacts::role_api) fn stage_apis(
    config: &ProtocolArtifactConfig,
    artifacts: &[ProtocolRustArtifact],
) -> Vec<StageRustApi> {
    artifacts
        .iter()
        .filter(|artifact| artifact.stage.is_proof())
        .map(|artifact| stage_api(config, artifact))
        .collect()
}

fn stage_api(config: &ProtocolArtifactConfig, artifact: &ProtocolRustArtifact) -> StageRustApi {
    let source = artifact.source.source.as_str();
    let module_name = artifact.stage.module_name();
    let role = RoleApiRole::from_role(&artifact.role);
    let type_inventory = discover_stage_types(source, module_name);
    let prefix = type_inventory.prefix.as_str();
    let program_binding = discover_program_binding(
        source,
        role.stage_program_type_suffix(),
        &["prove_", "execute_"],
    );
    let error_type = discover_stage_error_type(source, role, prefix);
    StageRustApi {
        field_name: module_name.to_owned(),
        module_alias: module_alias(module_name),
        variant_name: upper_camel(module_name),
        output_type: type_inventory.output_type,
        eval_type: type_inventory.eval_type,
        artifacts_type: type_inventory.artifacts_type,
        error_type,
        program_binding,
        kernel_module: find_kernel_module(config, source),
        opening_input_type: type_inventory.opening_input_type,
        ram_data_type: type_inventory.ram_data_type,
        verifier_data_type: type_inventory.verifier_data_type,
    }
}
