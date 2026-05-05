use super::super::super::role::RoleApiRole;
use super::super::super::{CommitmentRustApi, StageRustApi};

pub(in crate::emit::rust::artifacts::role_api) fn push_prover_error_conversions(
    source: &mut String,
    commitment: Option<&CommitmentRustApi>,
    stages: &[StageRustApi],
    prove_error_type: &str,
) {
    if let Some(commitment) = commitment {
        source.push_str(&format!(
            "impl From<{module}::{error}> for {prove_error_type} {{\n    fn from(error: {module}::{error}) -> Self {{\n        Self::{variant}(error)\n    }}\n}}\n\n",
            module = commitment.module_alias,
            error = commitment.error_type,
            variant = commitment.variant_name,
        ));
    }
    for stage in stages {
        let module = RoleApiRole::Prover.stage_module_alias(stage);
        source.push_str(&format!(
            "impl From<{module}::{error}> for {prove_error_type} {{\n    fn from(error: {module}::{error}) -> Self {{\n        Self::{variant}(error)\n    }}\n}}\n\n",
            error = stage.error_type,
            variant = stage.variant_name,
        ));
    }
}
