use super::super::super::role::RoleApiRole;
use super::super::super::{CommitmentRustApi, StageRustApi};

pub(in crate::emit::rust::artifacts::role_api) fn push_error_variants(
    source: &mut String,
    commitment: Option<&CommitmentRustApi>,
    stages: &[StageRustApi],
    role: RoleApiRole,
) {
    if let Some(commitment) = commitment {
        source.push_str(&format!(
            "    {}({}::{}),\n",
            commitment.variant_name, commitment.module_alias, commitment.error_type
        ));
    }
    for stage in stages {
        source.push_str(&format!(
            "    {}({}::{}),\n",
            stage.variant_name,
            role.stage_module_alias(stage),
            stage.error_type
        ));
    }
}
