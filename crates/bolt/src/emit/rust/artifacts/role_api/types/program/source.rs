use super::super::super::role::RoleApiRole;
use super::RoleApiProgramBinding;

pub(in crate::emit::rust::artifacts::role_api) trait RoleApiProgramSource {
    fn field_name(&self) -> &str;
    fn program_binding(&self) -> &RoleApiProgramBinding;

    fn program_argument(&self, role: RoleApiRole) -> Option<String> {
        self.program_binding()
            .program(role)
            .is_some()
            .then(|| format!("programs.{}", self.field_name()))
    }

    fn program_argument_prefix(&self, role: RoleApiRole) -> String {
        self.program_argument(role)
            .map(|argument| format!("{argument}, "))
            .unwrap_or_default()
    }

    fn program(&self, role: RoleApiRole) -> Option<super::RoleApiProgram<'_>> {
        self.program_binding().program(role)
    }

    fn prover_entrypoint(&self) -> Option<&str> {
        self.program_binding().entrypoint(RoleApiRole::Prover)
    }

    fn verifier_entrypoint(&self) -> Option<&str> {
        self.program_binding().entrypoint(RoleApiRole::Verifier)
    }
}
