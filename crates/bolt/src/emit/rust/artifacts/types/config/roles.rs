use crate::ir::Role;

use super::super::generated::ProtocolRuntimeModule;
use super::ProtocolArtifactConfig;

impl ProtocolArtifactConfig {
    pub(in crate::emit::rust::artifacts) fn crate_name(&self, role: &Role) -> &str {
        match role {
            Role::Prover => &self.prover_crate_name,
            Role::Verifier => &self.verifier_crate_name,
        }
    }

    pub(in crate::emit::rust::artifacts) fn forbidden_imports(&self, role: &Role) -> &[String] {
        match role {
            Role::Prover => &self.prover_forbidden_imports,
            Role::Verifier => &self.verifier_forbidden_imports,
        }
    }

    pub(in crate::emit::rust::artifacts) fn runtime_modules(
        &self,
        role: &Role,
    ) -> &[ProtocolRuntimeModule] {
        match role {
            Role::Prover => &[],
            Role::Verifier => &self.verifier_runtime_modules,
        }
    }
}
