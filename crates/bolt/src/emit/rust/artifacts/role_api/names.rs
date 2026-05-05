use super::role::RoleApiRole;

pub(super) struct RoleApiNames {
    pub(super) named_eval: String,
    pub(super) sumcheck_output: String,
    pub(super) stage_proof: String,
    pub(super) proof: String,
    pub(super) prover_inputs: String,
    pub(super) prover_programs: String,
    pub(super) prover_artifacts: String,
    pub(super) prove_error: String,
    pub(super) default_transcript_alias: String,
    pub(super) verifier_inputs: String,
    pub(super) verifier_programs: String,
    pub(super) verification_artifacts: String,
    pub(super) verify_error: String,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RoleApiRoleNames<'a> {
    pub(super) inputs: &'a str,
    pub(super) programs: &'a str,
    pub(super) artifacts: &'a str,
    pub(super) error: &'a str,
}

impl RoleApiNames {
    pub(super) fn new(prefix: &str) -> Self {
        Self {
            named_eval: format!("{prefix}NamedEval"),
            sumcheck_output: format!("{prefix}SumcheckOutput"),
            stage_proof: format!("{prefix}StageProof"),
            proof: format!("{prefix}Proof"),
            prover_inputs: format!("{prefix}ProverInputs"),
            prover_programs: format!("{prefix}ProverPrograms"),
            prover_artifacts: format!("{prefix}ProverArtifacts"),
            prove_error: format!("{prefix}ProveError"),
            default_transcript_alias: format!("Default{prefix}Transcript"),
            verifier_inputs: format!("{prefix}VerifierInputs"),
            verifier_programs: format!("{prefix}VerifierPrograms"),
            verification_artifacts: format!("{prefix}VerificationArtifacts"),
            verify_error: format!("{prefix}VerifyError"),
        }
    }

    pub(super) fn role(&self, role: RoleApiRole) -> RoleApiRoleNames<'_> {
        match role {
            RoleApiRole::Prover => RoleApiRoleNames {
                inputs: &self.prover_inputs,
                programs: &self.prover_programs,
                artifacts: &self.prover_artifacts,
                error: &self.prove_error,
            },
            RoleApiRole::Verifier => RoleApiRoleNames {
                inputs: &self.verifier_inputs,
                programs: &self.verifier_programs,
                artifacts: &self.verification_artifacts,
                error: &self.verify_error,
            },
        }
    }
}
