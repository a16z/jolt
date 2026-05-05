mod extension;

use crate::ir::Role;

use super::types::StageRustApi;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RoleApiRole {
    Prover,
    Verifier,
}

impl RoleApiRole {
    pub(super) fn from_role(role: &Role) -> Self {
        match role {
            Role::Prover => Self::Prover,
            Role::Verifier => Self::Verifier,
        }
    }

    pub(super) fn source_path(self) -> &'static str {
        match self {
            Self::Prover => "src/prover.rs",
            Self::Verifier => "src/verifier.rs",
        }
    }

    pub(super) fn stage_program_type_suffix(self) -> &'static str {
        match self {
            Self::Prover => "CpuProgramPlan",
            Self::Verifier => "VerifierProgramPlan",
        }
    }

    pub(super) fn commitment_program_type_suffix(self) -> &'static str {
        match self {
            Self::Prover => "ProverProgramPlan",
            Self::Verifier => "VerifierProgramPlan",
        }
    }

    pub(super) fn default_programs_fn_name(self) -> &'static str {
        match self {
            Self::Prover => "default_prover_programs",
            Self::Verifier => "default_verifier_programs",
        }
    }

    pub(super) fn stage_module_alias<'a>(self, stage: &'a StageRustApi) -> &'a str {
        match self {
            Self::Prover => stage
                .kernel_module
                .as_deref()
                .unwrap_or(stage.module_alias.as_str()),
            Self::Verifier => stage.module_alias.as_str(),
        }
    }
}
