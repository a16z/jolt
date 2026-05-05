pub(in crate::emit::rust::artifacts::role_api::verifier) struct VerifierEntryTypes<'a> {
    pub protocol_snake: &'a str,
    pub field_type: &'a str,
    pub transcript_trait: &'a str,
    pub proof_type: &'a str,
    pub verifier_inputs_type: &'a str,
    pub verifier_programs_type: &'a str,
    pub verification_artifacts_type: &'a str,
    pub verify_error_type: &'a str,
    pub instrumentation_prefix: Option<&'a str>,
}
