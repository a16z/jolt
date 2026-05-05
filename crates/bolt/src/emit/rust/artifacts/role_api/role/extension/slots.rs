use super::super::super::super::types::{ProtocolProverApiExtension, ProtocolVerifierApiExtension};

pub(super) struct RoleExtensionSlots<'a> {
    pub(super) program_fields: &'a str,
    pub(super) default_program_fields: &'a str,
    pub(super) helper_items: &'a str,
    pub(super) error_variants: &'a str,
    pub(super) error_items: &'a str,
    pub(super) error_conversions: &'a str,
    pub(super) imports: &'a str,
    pub(super) lib_module: &'a str,
    pub(super) input_fields: &'a str,
    pub(super) proof_fields: &'a str,
}

impl<'a> RoleExtensionSlots<'a> {
    pub(super) fn from_prover(extension: &'a ProtocolProverApiExtension) -> Self {
        Self {
            program_fields: extension.program_fields.as_str(),
            default_program_fields: extension.default_program_fields.as_str(),
            helper_items: extension.helper_items.as_str(),
            error_variants: extension.error_variants.as_str(),
            error_items: extension.error_items.as_str(),
            error_conversions: extension.error_conversions.as_str(),
            imports: extension.imports.as_str(),
            lib_module: extension.lib_module.as_str(),
            input_fields: extension.input_fields.as_str(),
            proof_fields: extension.proof_fields.as_str(),
        }
    }

    pub(super) fn from_verifier(extension: &'a ProtocolVerifierApiExtension) -> Self {
        Self {
            program_fields: extension.program_fields.as_str(),
            default_program_fields: extension.default_program_fields.as_str(),
            helper_items: extension.helper_items.as_str(),
            error_variants: extension.error_variants.as_str(),
            error_items: extension.error_items.as_str(),
            error_conversions: extension.error_conversions.as_str(),
            imports: extension.imports.as_str(),
            lib_module: extension.lib_module.as_str(),
            input_fields: extension.input_fields.as_str(),
            proof_fields: extension.proof_fields.as_str(),
        }
    }
}
