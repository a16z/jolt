use super::super::super::super::types::ProtocolArtifactExtension;
use super::super::RoleApiRole;

impl RoleApiRole {
    pub(in crate::emit::rust::artifacts::role_api) fn extension_program_fields<'a>(
        self,
        extension: &'a ProtocolArtifactExtension,
    ) -> &'a str {
        self.extension_slots(extension).program_fields
    }

    pub(in crate::emit::rust::artifacts::role_api) fn extension_default_program_fields<'a>(
        self,
        extension: &'a ProtocolArtifactExtension,
    ) -> &'a str {
        self.extension_slots(extension).default_program_fields
    }
}
