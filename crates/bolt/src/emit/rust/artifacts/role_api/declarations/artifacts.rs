use super::super::items::push_artifact_fields;
use super::super::role::RoleApiRole;
use super::super::{CommitmentRustApi, StageRustApi};

pub(in crate::emit::rust::artifacts::role_api) fn push_artifacts_struct(
    source: &mut String,
    artifacts_type: &str,
    commitment: Option<&CommitmentRustApi>,
    stages: &[StageRustApi],
    field_type: &str,
    role: RoleApiRole,
) {
    source.push_str(&format!(
        "#[derive(Clone, Debug)]\npub struct {artifacts_type} {{\n"
    ));
    push_artifact_fields(source, commitment, stages, field_type, role);
    source.push_str("}\n\n");
}
