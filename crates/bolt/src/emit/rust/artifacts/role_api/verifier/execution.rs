mod calls;

use super::super::super::types::ProtocolArtifactExtension;
use super::super::items::push_artifact_values;
use super::super::{CommitmentRustApi, StageRustApi};
use calls::{push_commitment_execution, push_stage_execution};

pub(super) fn push_with_programs_body(
    source: &mut String,
    commitment: Option<&CommitmentRustApi>,
    stages: &[StageRustApi],
    verification_artifacts_type: &str,
    extension: Option<&ProtocolArtifactExtension>,
) {
    source.push_str("{\n");
    if let Some(extension) = extension {
        source.push_str(&extension.verifier.with_programs_body_intro);
    }
    if let Some(commitment) = commitment {
        push_commitment_execution(source, commitment);
    }
    for stage in stages {
        push_stage_execution(source, stage);
    }
    if let Some(extension) = extension {
        source.push_str(&extension.verifier.after_stage_verification);
    }
    source.push_str(&format!("\n    Ok({verification_artifacts_type} {{\n"));
    push_artifact_values(source, commitment, stages);
    source.push_str("    })\n}\n\n");
}
