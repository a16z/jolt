use super::super::super::role::RoleApiRole;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::emit::rust::artifacts::role_api) struct RoleApiProgram<'a> {
    pub(in crate::emit::rust::artifacts::role_api) type_name: &'a str,
    pub(in crate::emit::rust::artifacts::role_api) const_name: &'a str,
}

#[derive(Clone, Debug, Default)]
pub(in crate::emit::rust::artifacts::role_api) struct RoleApiProgramBinding {
    pub(in crate::emit::rust::artifacts::role_api) verifier_fn: Option<String>,
    pub(in crate::emit::rust::artifacts::role_api) with_program_verifier_fn: Option<String>,
    pub(in crate::emit::rust::artifacts::role_api) program_type: Option<String>,
    pub(in crate::emit::rust::artifacts::role_api) program_const: Option<String>,
    pub(in crate::emit::rust::artifacts::role_api) prover_fn: Option<String>,
    pub(in crate::emit::rust::artifacts::role_api) with_program_prover_fn: Option<String>,
}

impl RoleApiProgramBinding {
    pub(in crate::emit::rust::artifacts::role_api) fn program(
        &self,
        role: RoleApiRole,
    ) -> Option<RoleApiProgram<'_>> {
        let _ = self.with_program_entrypoint(role)?;
        Some(RoleApiProgram {
            type_name: self.program_type.as_deref()?,
            const_name: self.program_const.as_deref()?,
        })
    }

    pub(in crate::emit::rust::artifacts::role_api) fn entrypoint(
        &self,
        role: RoleApiRole,
    ) -> Option<&str> {
        self.with_program_entrypoint(role)
            .or_else(|| self.default_entrypoint(role))
    }

    fn with_program_entrypoint(&self, role: RoleApiRole) -> Option<&str> {
        match role {
            RoleApiRole::Prover => self.with_program_prover_fn.as_deref(),
            RoleApiRole::Verifier => self.with_program_verifier_fn.as_deref(),
        }
    }

    fn default_entrypoint(&self, role: RoleApiRole) -> Option<&str> {
        match role {
            RoleApiRole::Prover => self.prover_fn.as_deref(),
            RoleApiRole::Verifier => self.verifier_fn.as_deref(),
        }
    }
}
