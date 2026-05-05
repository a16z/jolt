use super::program::{RoleApiProgramBinding, RoleApiProgramSource};

#[derive(Clone, Debug)]
pub(in crate::emit::rust::artifacts::role_api) struct CommitmentRustApi {
    pub(in crate::emit::rust::artifacts::role_api) field_name: String,
    pub(in crate::emit::rust::artifacts::role_api) module_alias: String,
    pub(in crate::emit::rust::artifacts::role_api) variant_name: String,
    pub(in crate::emit::rust::artifacts::role_api) artifacts_type: String,
    pub(in crate::emit::rust::artifacts::role_api) error_type: String,
    pub(in crate::emit::rust::artifacts::role_api) program_binding: RoleApiProgramBinding,
    pub(in crate::emit::rust::artifacts::role_api) input_provider_trait: Option<String>,
}

impl RoleApiProgramSource for CommitmentRustApi {
    fn field_name(&self) -> &str {
        &self.field_name
    }

    fn program_binding(&self) -> &RoleApiProgramBinding {
        &self.program_binding
    }
}
