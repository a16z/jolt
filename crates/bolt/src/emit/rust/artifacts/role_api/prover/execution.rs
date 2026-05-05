mod calls;

use super::super::super::types::ProtocolArtifactExtension;
use super::super::items::push_artifact_values;
use super::super::role::RoleApiRole;
use super::super::{CommitmentRustApi, StageRustApi};
use calls::{push_commitment_execution, push_stage_execution};

const ROLE: RoleApiRole = RoleApiRole::Prover;

pub(super) fn push_with_programs_body(
    source: &mut String,
    commitment: Option<&CommitmentRustApi>,
    stages: &[StageRustApi],
    proof_type: &str,
    prover_artifacts_type: &str,
    extension: Option<&ProtocolArtifactExtension>,
) {
    source.push_str("{\n");
    if let Some(commitment) = commitment {
        push_commitment_execution(source, commitment);
    }
    for stage in stages {
        push_stage_execution(source, stage);
    }
    if let Some(extension) = extension {
        source.push_str(&extension.prover.after_stage_execution);
    }
    source.push_str(&format!("\n    let proof = {proof_type} {{\n"));
    if let Some(commitment) = commitment {
        source.push_str(&format!(
            "        commitments: {}.commitments.clone(),\n",
            commitment.field_name
        ));
    }
    for stage in stages {
        source.push_str(&format!(
            "        {}: {}_proof(&{}),\n",
            stage.field_name, stage.field_name, stage.field_name
        ));
    }
    if let Some(extension) = extension {
        source.push_str(ROLE.extension_proof_fields(extension));
    }
    source.push_str(&format!(
        "    }};\n    let artifacts = {prover_artifacts_type} {{\n"
    ));
    push_artifact_values(source, commitment, stages);
    source.push_str("    };\n    Ok((proof, artifacts))\n}\n\n");
}
