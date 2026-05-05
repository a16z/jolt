use super::super::role::RoleApiRole;
use super::super::{CommitmentRustApi, StageRustApi};

pub(in crate::emit::rust::artifacts::role_api) fn push_artifact_fields(
    source: &mut String,
    commitment: Option<&CommitmentRustApi>,
    stages: &[StageRustApi],
    field_type: &str,
    role: RoleApiRole,
) {
    if let Some(commitment) = commitment {
        source.push_str(&format!(
            "    pub {}: {}::{},\n",
            commitment.field_name, commitment.module_alias, commitment.artifacts_type
        ));
    }
    for stage in stages {
        source.push_str(&format!(
            "    pub {}: {}::{}<{field_type}>,\n",
            stage.field_name,
            role.stage_module_alias(stage),
            stage.artifacts_type
        ));
    }
}

pub(in crate::emit::rust::artifacts::role_api) fn push_artifact_values(
    source: &mut String,
    commitment: Option<&CommitmentRustApi>,
    stages: &[StageRustApi],
) {
    if let Some(commitment) = commitment {
        push_artifact_value(source, &commitment.field_name);
    }
    for stage in stages {
        push_artifact_value(source, &stage.field_name);
    }
}

fn push_artifact_value(source: &mut String, field_name: &str) {
    source.push_str(&format!("        {field_name},\n"));
}
