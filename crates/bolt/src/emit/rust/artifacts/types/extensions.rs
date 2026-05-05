#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolArtifactExtension {
    pub required_commitment: bool,
    pub required_proof_stages: Vec<String>,
    pub required_artifact_stages: Vec<String>,
    pub prover: ProtocolProverApiExtension,
    pub verifier: ProtocolVerifierApiExtension,
}

impl ProtocolArtifactExtension {
    pub(in crate::emit::rust::artifacts) fn is_active_with(
        &self,
        commitment_available: bool,
        mut proof_stage_available: impl FnMut(&str) -> bool,
        mut artifact_stage_available: impl FnMut(&str) -> bool,
    ) -> bool {
        (!self.required_commitment || commitment_available)
            && self
                .required_proof_stages
                .iter()
                .all(|required| proof_stage_available(required))
            && self
                .required_artifact_stages
                .iter()
                .all(|required| artifact_stage_available(required))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProtocolProverApiExtension {
    pub lib_module: String,
    pub imports: String,
    pub input_fields: String,
    pub program_fields: String,
    pub default_program_fields: String,
    pub error_variants: String,
    pub error_items: String,
    pub error_conversions: String,
    pub after_stage_execution: String,
    pub proof_fields: String,
    pub helper_items: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProtocolVerifierApiExtension {
    pub lib_module: String,
    pub imports: String,
    pub proof_fields: String,
    pub proof_items: String,
    pub inputs_derive: Option<String>,
    pub input_fields: String,
    pub program_fields: String,
    pub default_program_fields: String,
    pub error_variants: String,
    pub error_items: String,
    pub error_conversions: String,
    pub after_default_verify: String,
    pub with_programs_body_intro: String,
    pub after_stage_verification: String,
    pub helper_items: String,
}
