use super::super::super::super::types::ProtocolArtifactExtension;
use super::super::RoleApiRole;

impl RoleApiRole {
    pub(in crate::emit::rust::artifacts::role_api) fn extension_helper_items<'a>(
        self,
        extension: &'a ProtocolArtifactExtension,
    ) -> &'a str {
        self.extension_slots(extension).helper_items
    }

    pub(in crate::emit::rust::artifacts::role_api) fn extension_imports<'a>(
        self,
        extension: &'a ProtocolArtifactExtension,
    ) -> &'a str {
        self.extension_slots(extension).imports
    }

    pub(in crate::emit::rust::artifacts::role_api) fn extension_lib_module<'a>(
        self,
        extension: &'a ProtocolArtifactExtension,
    ) -> &'a str {
        self.extension_slots(extension).lib_module
    }

    pub(in crate::emit::rust::artifacts::role_api) fn extension_input_fields<'a>(
        self,
        extension: &'a ProtocolArtifactExtension,
    ) -> &'a str {
        self.extension_slots(extension).input_fields
    }

    pub(in crate::emit::rust::artifacts::role_api) fn extension_proof_fields<'a>(
        self,
        extension: &'a ProtocolArtifactExtension,
    ) -> &'a str {
        self.extension_slots(extension).proof_fields
    }
}
