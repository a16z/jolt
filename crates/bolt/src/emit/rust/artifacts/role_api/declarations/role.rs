use super::super::super::types::ProtocolArtifactExtension;
use super::super::role::RoleApiRole;
use super::super::{CommitmentRustApi, StageRustApi};
use super::{push_artifacts_struct, push_default_programs_fn, push_programs_struct};

pub(in crate::emit::rust::artifacts::role_api) struct RoleDeclarationTypes<'a> {
    pub(in crate::emit::rust::artifacts::role_api) programs: &'a str,
    pub(in crate::emit::rust::artifacts::role_api) artifacts: &'a str,
    pub(in crate::emit::rust::artifacts::role_api) field: &'a str,
}

pub(in crate::emit::rust::artifacts::role_api) fn push_role_program_artifact_declarations(
    source: &mut String,
    commitment: Option<&CommitmentRustApi>,
    stages: &[StageRustApi],
    role: RoleApiRole,
    types: RoleDeclarationTypes<'_>,
    extension: Option<&ProtocolArtifactExtension>,
) {
    push_programs_struct(
        source,
        types.programs,
        commitment,
        stages,
        role,
        extension.map(|extension| role.extension_program_fields(extension)),
    );

    push_default_programs_fn(
        source,
        types.programs,
        commitment,
        stages,
        role,
        extension.map(|extension| role.extension_default_program_fields(extension)),
    );

    push_artifacts_struct(
        source,
        types.artifacts,
        commitment,
        stages,
        types.field,
        role,
    );
}
