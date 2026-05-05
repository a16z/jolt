use super::super::super::super::types::ProtocolArtifactExtension;
use super::super::super::role::RoleApiRole;
use super::super::super::{StageRustApi, VerifierStageInputKind};

const ROLE: RoleApiRole = RoleApiRole::Verifier;

pub(in crate::emit::rust::artifacts::role_api::verifier) fn push_verifier_inputs(
    source: &mut String,
    stages: &[StageRustApi],
    field_type: &str,
    verifier_inputs_type: &str,
    extension: Option<&ProtocolArtifactExtension>,
) {
    let verifier_inputs_derive = extension
        .and_then(|extension| extension.verifier.inputs_derive.as_deref())
        .unwrap_or("#[derive(Clone, Copy, Debug)]");
    source.push_str(&format!(
        "{verifier_inputs_derive}\npub struct {verifier_inputs_type}<'a> {{\n"
    ));
    for stage in stages {
        for input in stage.verifier_inputs() {
            match input.kind {
                VerifierStageInputKind::Openings => {
                    source.push_str(&format!(
                        "    pub {}_{}: &'a [{}::{}<{field_type}>],\n",
                        stage.field_name,
                        input.kind.field_suffix(),
                        stage.module_alias,
                        input.type_name
                    ));
                }
                VerifierStageInputKind::Ram => {
                    source.push_str(&format!(
                        "    pub {}_{}: Option<&'a {}::{}<'a>>,\n",
                        stage.field_name,
                        input.kind.field_suffix(),
                        stage.module_alias,
                        input.type_name
                    ));
                }
                VerifierStageInputKind::Data => {
                    source.push_str(&format!(
                        "    pub {}_{}: Option<&'a {}::{}>,\n",
                        stage.field_name,
                        input.kind.field_suffix(),
                        stage.module_alias,
                        input.type_name
                    ));
                }
            }
        }
    }
    if let Some(extension) = extension {
        source.push_str(ROLE.extension_input_fields(extension));
    }
    source.push_str("}\n\n");
}
