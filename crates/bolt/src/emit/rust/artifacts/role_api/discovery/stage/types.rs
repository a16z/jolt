use super::super::super::super::support::upper_camel;
use super::super::super::source_scan::{find_type_with_suffix, has_public_type_name};

pub(super) struct StageTypeInventory {
    pub(super) prefix: String,
    pub(super) artifacts_type: String,
    pub(super) output_type: String,
    pub(super) eval_type: String,
    pub(super) opening_input_type: Option<String>,
    pub(super) ram_data_type: Option<String>,
    pub(super) verifier_data_type: Option<String>,
}

pub(super) fn discover_stage_types(source: &str, module_name: &str) -> StageTypeInventory {
    let artifacts_type = find_type_with_suffix(source, "ExecutionArtifacts")
        .unwrap_or_else(|| format!("{}ExecutionArtifacts", upper_camel(module_name)));
    let prefix = artifacts_type
        .strip_suffix("ExecutionArtifacts")
        .unwrap_or(&artifacts_type)
        .to_owned();
    let opening_input_name = format!("{prefix}OpeningInputValue");
    let ram_data_name = format!("{prefix}RamData");
    let verifier_data_name = format!("{prefix}VerifierData");
    StageTypeInventory {
        output_type: format!("{prefix}SumcheckOutput"),
        eval_type: format!("{prefix}NamedEval"),
        artifacts_type,
        prefix,
        opening_input_type: has_public_type_name(source, &opening_input_name)
            .then_some(opening_input_name),
        ram_data_type: has_public_type_name(source, &ram_data_name).then_some(ram_data_name),
        verifier_data_type: has_public_type_name(source, &verifier_data_name)
            .then_some(verifier_data_name),
    }
}
