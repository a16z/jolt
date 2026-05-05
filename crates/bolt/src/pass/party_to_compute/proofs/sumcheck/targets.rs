use crate::ir::Role;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RoleSumcheckTargets {
    pub(super) claim_op: &'static str,
    pub(super) driver_op: &'static str,
}

impl RoleSumcheckTargets {
    pub(super) fn for_role(role: &Role) -> Self {
        match role {
            Role::Prover => Self {
                claim_op: "compute.sumcheck_claim",
                driver_op: "compute.sumcheck_driver",
            },
            Role::Verifier => Self {
                claim_op: "compute.sumcheck_verify_claim",
                driver_op: "compute.sumcheck_verify",
            },
        }
    }
}
