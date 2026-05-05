use super::super::items::push_error_variants;
use super::super::role::RoleApiRole;
use super::super::{CommitmentRustApi, StageRustApi};

pub(in crate::emit::rust::artifacts::role_api) fn push_error_enum(
    source: &mut String,
    error_type: &str,
    commitment: Option<&CommitmentRustApi>,
    stages: &[StageRustApi],
    role: RoleApiRole,
    extension_variants: Option<&str>,
) {
    source.push_str(&format!("#[derive(Debug)]\npub enum {error_type} {{\n"));
    push_error_variants(source, commitment, stages, role);
    if let Some(extension_variants) = extension_variants {
        source.push_str(extension_variants);
    }
    source.push_str("}\n\n");
}
