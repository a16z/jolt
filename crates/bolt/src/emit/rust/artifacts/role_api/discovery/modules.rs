use super::super::super::types::ProtocolRustArtifact;
use super::super::{CommitmentRustApi, StageRustApi};

pub(in crate::emit::rust::artifacts::role_api) fn role_modules(
    artifacts: &[ProtocolRustArtifact],
) -> Vec<String> {
    artifacts
        .iter()
        .map(|artifact| artifact.stage.module_name().to_owned())
        .collect()
}

pub(in crate::emit::rust::artifacts::role_api) fn aliased_modules(
    modules: &[String],
) -> Vec<String> {
    modules
        .iter()
        .map(|module| format!("{module} as {}", module_alias(module)))
        .collect()
}

pub(in crate::emit::rust::artifacts::role_api) fn module_alias(module: &str) -> String {
    format!("{module}_stage")
}

pub(in crate::emit::rust::artifacts::role_api) fn unique_kernel_modules(
    stages: &[StageRustApi],
) -> Vec<String> {
    let mut modules = Vec::new();
    for stage in stages {
        if let Some(kernel_module) = &stage.kernel_module {
            if !modules.contains(kernel_module) {
                modules.push(kernel_module.clone());
            }
        }
    }
    modules
}

pub(in crate::emit::rust::artifacts::role_api) fn prover_generic_params(
    stages: &[StageRustApi],
    commitment: Option<&CommitmentRustApi>,
) -> Vec<String> {
    let mut params = if commitment.is_some() {
        vec!["CommitmentInputs".to_owned()]
    } else {
        Vec::new()
    };
    params.extend(
        stages
            .iter()
            .map(|stage| format!("{}Executor", stage.variant_name)),
    );
    params
}

pub(in crate::emit::rust::artifacts::role_api) fn kernel_executor_type(error_type: &str) -> String {
    error_type.strip_suffix("KernelError").map_or_else(
        || {
            error_type
                .replace("Verify", "")
                .replace("Error", "KernelExecutor")
        },
        |prefix| format!("{prefix}KernelExecutor"),
    )
}
