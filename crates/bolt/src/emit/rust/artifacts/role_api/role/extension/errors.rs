use super::super::super::super::types::ProtocolArtifactExtension;
use super::super::RoleApiRole;

impl RoleApiRole {
    pub(in crate::emit::rust::artifacts::role_api) fn extension_error_variants<'a>(
        self,
        extension: &'a ProtocolArtifactExtension,
    ) -> &'a str {
        self.extension_slots(extension).error_variants
    }

    pub(in crate::emit::rust::artifacts::role_api) fn extension_error_items<'a>(
        self,
        extension: &'a ProtocolArtifactExtension,
    ) -> &'a str {
        self.extension_slots(extension).error_items
    }

    pub(in crate::emit::rust::artifacts::role_api) fn extension_error_conversions<'a>(
        self,
        extension: &'a ProtocolArtifactExtension,
    ) -> &'a str {
        self.extension_slots(extension).error_conversions
    }
}
