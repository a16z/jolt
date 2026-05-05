use super::program::{RoleApiProgramBinding, RoleApiProgramSource};
use super::verifier_input::{VerifierStageInput, VerifierStageInputKind};

#[derive(Clone, Debug)]
pub(in crate::emit::rust::artifacts::role_api) struct StageRustApi {
    pub(in crate::emit::rust::artifacts::role_api) field_name: String,
    pub(in crate::emit::rust::artifacts::role_api) module_alias: String,
    pub(in crate::emit::rust::artifacts::role_api) variant_name: String,
    pub(in crate::emit::rust::artifacts::role_api) output_type: String,
    pub(in crate::emit::rust::artifacts::role_api) eval_type: String,
    pub(in crate::emit::rust::artifacts::role_api) artifacts_type: String,
    pub(in crate::emit::rust::artifacts::role_api) error_type: String,
    pub(in crate::emit::rust::artifacts::role_api) program_binding: RoleApiProgramBinding,
    pub(in crate::emit::rust::artifacts::role_api) kernel_module: Option<String>,
    pub(in crate::emit::rust::artifacts::role_api) opening_input_type: Option<String>,
    pub(in crate::emit::rust::artifacts::role_api) ram_data_type: Option<String>,
    pub(in crate::emit::rust::artifacts::role_api) verifier_data_type: Option<String>,
}

impl StageRustApi {
    pub(in crate::emit::rust::artifacts::role_api) fn verifier_inputs(
        &self,
    ) -> Vec<VerifierStageInput<'_>> {
        let mut inputs = Vec::new();
        if let Some(type_name) = self.opening_input_type.as_deref() {
            inputs.push(VerifierStageInput {
                kind: VerifierStageInputKind::Openings,
                type_name,
            });
        }
        if let Some(type_name) = self.ram_data_type.as_deref() {
            inputs.push(VerifierStageInput {
                kind: VerifierStageInputKind::Ram,
                type_name,
            });
        }
        if let Some(type_name) = self.verifier_data_type.as_deref() {
            inputs.push(VerifierStageInput {
                kind: VerifierStageInputKind::Data,
                type_name,
            });
        }
        inputs
    }
}

impl RoleApiProgramSource for StageRustApi {
    fn field_name(&self) -> &str {
        &self.field_name
    }

    fn program_binding(&self) -> &RoleApiProgramBinding {
        &self.program_binding
    }
}
