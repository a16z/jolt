use super::super::programs::{push_default_program_fields, push_program_struct_fields};
use super::super::role::RoleApiRole;
use super::super::{CommitmentRustApi, StageRustApi};

pub(in crate::emit::rust::artifacts::role_api) fn push_programs_struct(
    source: &mut String,
    programs_type: &str,
    commitment: Option<&CommitmentRustApi>,
    stages: &[StageRustApi],
    role: RoleApiRole,
    extension_fields: Option<&str>,
) {
    source.push_str(&format!(
        "#[derive(Clone, Copy, Debug)]\npub struct {programs_type} {{\n"
    ));
    push_program_struct_fields(source, commitment, stages, role);
    if let Some(extension_fields) = extension_fields {
        source.push_str(extension_fields);
    }
    source.push_str("}\n\n");
}

pub(in crate::emit::rust::artifacts::role_api) fn push_default_programs_fn(
    source: &mut String,
    programs_type: &str,
    commitment: Option<&CommitmentRustApi>,
    stages: &[StageRustApi],
    role: RoleApiRole,
    extension_fields: Option<&str>,
) {
    let fn_name = role.default_programs_fn_name();
    source.push_str(&format!(
        "pub fn {fn_name}() -> {programs_type} {{\n    {programs_type} {{\n"
    ));
    push_default_program_fields(source, commitment, stages, role);
    if let Some(extension_fields) = extension_fields {
        source.push_str(extension_fields);
    }
    source.push_str("    }\n}\n\n");
}
