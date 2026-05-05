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
    instrumentation_prefix: Option<&str>,
) {
    source.push_str("{\n");
    if let Some(extension) = extension {
        source.push_str(&extension.verifier.with_programs_body_intro);
    }
    if let Some(prefix) = instrumentation_prefix {
        source.push_str(&format!(
            "    let _verify_span = tracing::info_span!(\"{prefix}.verify\").entered();\n"
        ));
    }
    if let Some(commitment) = commitment {
        push_commitment_execution(source, commitment);
    }
    if let Some(extension) = extension {
        if !extension.verifier.stage_verification_override.is_empty() {
            source.push_str(&extension.verifier.stage_verification_override);
        } else {
            push_stage_executions(source, stages);
        }
    } else {
        push_stage_executions(source, stages);
    }
    if let Some(extension) = extension {
        source.push_str(&extension.verifier.after_stage_verification);
    }
    source.push_str(&format!("\n    Ok({verification_artifacts_type} {{\n"));
    push_artifact_values(source, commitment, stages);
    source.push_str("    })\n}\n\n");
}

fn push_stage_executions(source: &mut String, stages: &[StageRustApi]) {
    for stage in stages {
        push_stage_execution(source, stage);
    }
}
