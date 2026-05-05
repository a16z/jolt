use super::super::super::types::ProtocolArtifactExtension;
use super::super::role::RoleApiRole;
use super::super::{CommitmentRustApi, StageRustApi};

const ROLE: RoleApiRole = RoleApiRole::Prover;

pub(super) fn push_prover_inputs(
    source: &mut String,
    stages: &[StageRustApi],
    commitment: Option<&CommitmentRustApi>,
    prover_setup_type: &str,
    prover_inputs_type: &str,
    generic_params: &[String],
    extension: Option<&ProtocolArtifactExtension>,
) {
    source.push_str(&format!(
        "pub struct {prover_inputs_type}<'a, {}> {{\n",
        generic_params.join(", ")
    ));
    if commitment.is_some() {
        source.push_str("    pub commitment_inputs: &'a mut CommitmentInputs,\n");
        source.push_str(&format!("    pub prover_setup: &'a {prover_setup_type},\n"));
    }
    for stage in stages {
        source.push_str(&format!(
            "    pub {}_executor: &'a mut {}Executor,\n",
            stage.field_name, stage.variant_name
        ));
    }
    if let Some(extension) = extension {
        source.push_str(ROLE.extension_input_fields(extension));
    }
    source.push_str("}\n\n");
}
