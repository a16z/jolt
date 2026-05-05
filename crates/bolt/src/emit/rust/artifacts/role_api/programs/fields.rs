mod style;

use super::super::role::RoleApiRole;
use super::super::{CommitmentRustApi, RoleApiProgramSource, StageRustApi};
use style::ProgramFieldStyle;

pub(in crate::emit::rust::artifacts::role_api) fn push_program_struct_fields(
    source: &mut String,
    commitment: Option<&CommitmentRustApi>,
    stages: &[StageRustApi],
    role: RoleApiRole,
) {
    push_program_fields(source, commitment, stages, role, ProgramFieldStyle::Struct);
}

pub(in crate::emit::rust::artifacts::role_api) fn push_default_program_fields(
    source: &mut String,
    commitment: Option<&CommitmentRustApi>,
    stages: &[StageRustApi],
    role: RoleApiRole,
) {
    push_program_fields(source, commitment, stages, role, ProgramFieldStyle::Default);
}

fn push_program_fields(
    source: &mut String,
    commitment: Option<&CommitmentRustApi>,
    stages: &[StageRustApi],
    role: RoleApiRole,
    style: ProgramFieldStyle,
) {
    if let Some(commitment) = commitment {
        if let Some(program) = commitment.program(role) {
            style.push(
                source,
                &commitment.field_name,
                &commitment.module_alias,
                program,
            );
        }
    }
    for stage in stages {
        if let Some(program) = stage.program(role) {
            style.push(
                source,
                &stage.field_name,
                style.stage_module_alias(stage, role),
                program,
            );
        }
    }
}
