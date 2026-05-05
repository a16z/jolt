mod inventory;
mod role_exports;

use crate::ir::Role;

use super::super::support::byte_string_literal;
use super::super::types::{ProtocolArtifactConfig, ProtocolRustArtifact};
use super::discovery::{active_role_api_extension, commitment_api, stage_apis};
use super::names::RoleApiNames;
use inventory::generated_stage_inventory;
use role_exports::role_module_source;

pub(in crate::emit::rust::artifacts) fn generated_lib(
    config: &ProtocolArtifactConfig,
    role: &Role,
    artifacts: &[ProtocolRustArtifact],
) -> String {
    let protocol_snake = config.protocol_snake();
    let names = RoleApiNames::new(&config.type_prefix);
    let stage_apis = stage_apis(config, artifacts);
    let commitment_api = commitment_api(artifacts);
    let extension = active_role_api_extension(config, &stage_apis, &commitment_api, artifacts);
    let role_module = role_module_source(role, extension, &protocol_snake, &names);
    let stages = generated_stage_inventory(artifacts);
    format!(
        "{role_module}\n\npub const TRANSCRIPT_LABEL: &[u8] = {};\n\n#[derive(Clone, Copy, Debug, PartialEq, Eq)]\npub struct GeneratedStage {{\n    pub name: &'static str,\n    pub module: &'static str,\n    pub ordinal: usize,\n}}\n\npub const GENERATED_STAGES: &[GeneratedStage] = &[\n{stages}\n];\n\npub fn generated_stage_names() -> impl Iterator<Item = &'static str> {{\n    GENERATED_STAGES.iter().map(|stage| stage.name)\n}}\n",
        byte_string_literal(&config.transcript_label)
    )
}
