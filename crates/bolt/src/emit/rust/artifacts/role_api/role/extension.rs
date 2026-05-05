mod errors;
mod programs;
mod slots;
mod source;

use super::super::super::types::ProtocolArtifactExtension;
use super::RoleApiRole;
use slots::RoleExtensionSlots;

impl RoleApiRole {
    fn extension_slots<'a>(
        self,
        extension: &'a ProtocolArtifactExtension,
    ) -> RoleExtensionSlots<'a> {
        match self {
            Self::Prover => RoleExtensionSlots::from_prover(&extension.prover),
            Self::Verifier => RoleExtensionSlots::from_verifier(&extension.verifier),
        }
    }
}
