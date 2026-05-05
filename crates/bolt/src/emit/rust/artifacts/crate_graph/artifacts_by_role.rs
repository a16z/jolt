use crate::ir::Role;

use super::super::types::ProtocolRustArtifact;

#[derive(Default)]
pub(super) struct ArtifactsByRole {
    prover: Vec<ProtocolRustArtifact>,
    verifier: Vec<ProtocolRustArtifact>,
}

impl ArtifactsByRole {
    pub(super) fn push(&mut self, artifact: ProtocolRustArtifact) {
        match artifact.role {
            Role::Prover => self.prover.push(artifact),
            Role::Verifier => self.verifier.push(artifact),
        }
    }

    pub(super) fn into_role_artifacts(self) -> Vec<(Role, Vec<ProtocolRustArtifact>)> {
        vec![(Role::Prover, self.prover), (Role::Verifier, self.verifier)]
    }
}
