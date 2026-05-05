use super::super::super::discovery::kernel_executor_type;
use super::super::super::{CommitmentRustApi, StageRustApi};

pub(super) fn push_where_bounds(
    source: &mut String,
    commitment: Option<&CommitmentRustApi>,
    stages: &[StageRustApi],
    field_type: &str,
    transcript_trait: &str,
) {
    if let Some(commitment) = commitment {
        let input_provider = commitment
            .input_provider_trait
            .as_deref()
            .unwrap_or("MissingCommitmentInputProvider");
        source.push_str(&format!(
            "    CommitmentInputs: {}::{input_provider},\n",
            commitment.module_alias
        ));
    }
    for stage in stages {
        let kernel_module = stage
            .kernel_module
            .as_deref()
            .unwrap_or(stage.module_alias.as_str());
        let kernel_trait = kernel_executor_type(&stage.error_type);
        source.push_str(&format!(
            "    {}Executor: {}::{}<{field_type}>,\n",
            stage.variant_name, kernel_module, kernel_trait
        ));
    }
    source.push_str(&format!(
        "    T: {transcript_trait}<Challenge = {field_type}>,\n"
    ));
}
