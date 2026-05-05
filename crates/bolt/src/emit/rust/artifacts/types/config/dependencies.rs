use crate::ir::Role;

use super::ProtocolArtifactConfig;

impl ProtocolArtifactConfig {
    pub(in crate::emit::rust::artifacts) fn dependencies(&self, role: &Role) -> Vec<String> {
        let mut dependencies = self.common_dependencies.clone();
        match role {
            Role::Prover => {
                dependencies.extend(self.prover_dependencies.clone());
                if !dependencies.contains(&self.verifier_crate_name) {
                    dependencies.push(self.verifier_crate_name.clone());
                }
            }
            Role::Verifier => dependencies.extend(self.verifier_dependencies.clone()),
        }
        dependencies.sort();
        dependencies.dedup();
        dependencies
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolStandaloneDependency {
    pub package: String,
    pub manifest_entry: String,
}

impl ProtocolStandaloneDependency {
    pub fn new(package: impl Into<String>, manifest_entry: impl Into<String>) -> Self {
        Self {
            package: package.into(),
            manifest_entry: manifest_entry.into(),
        }
    }
}
