use super::super::super::super::types::ProtocolArtifactExtension;
use super::super::super::role::RoleApiRole;
use super::super::super::{CommitmentRustApi, StageRustApi};

const ROLE: RoleApiRole = RoleApiRole::Verifier;

pub(in crate::emit::rust::artifacts::role_api::verifier) fn push_proof_type(
    source: &mut String,
    commitment: Option<&CommitmentRustApi>,
    stages: &[StageRustApi],
    commitment_type: &str,
    proof_type: &str,
    stage_proof_type: &str,
    extension: Option<&ProtocolArtifactExtension>,
) {
    source.push_str(&format!(
        "#[derive(Clone, Debug)]\npub struct {proof_type} {{\n"
    ));
    if commitment.is_some() {
        source.push_str(&format!(
            "    pub commitments: Vec<Option<{commitment_type}>>,\n"
        ));
    }
    for stage in stages {
        source.push_str(&format!(
            "    pub {}: {stage_proof_type},\n",
            stage.field_name
        ));
    }
    if let Some(extension) = extension {
        source.push_str(ROLE.extension_proof_fields(extension));
    }
    source.push_str("}\n\n");

    if let Some(extension) = extension {
        source.push_str(&extension.verifier.proof_items);
    }
}
