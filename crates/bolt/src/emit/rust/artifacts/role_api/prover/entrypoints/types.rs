pub(in crate::emit::rust::artifacts::role_api::prover) struct ProverEntryTypes<'a> {
    pub protocol_snake: &'a str,
    pub generic_params: &'a [String],
    pub field_type: &'a str,
    pub transcript_trait: &'a str,
    pub proof_type: &'a str,
    pub prover_inputs_type: &'a str,
    pub prover_programs_type: &'a str,
    pub prover_artifacts_type: &'a str,
    pub prove_error_type: &'a str,
    pub instrumentation_prefix: Option<&'a str>,
}
