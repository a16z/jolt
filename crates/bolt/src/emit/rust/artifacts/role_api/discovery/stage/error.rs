use super::super::super::role::RoleApiRole;
use super::super::super::source_scan::{find_public_item, find_type_with_suffix};

pub(super) fn discover_stage_error_type(source: &str, role: RoleApiRole, prefix: &str) -> String {
    match role {
        RoleApiRole::Prover => find_type_with_suffix(source, "KernelError")
            .unwrap_or_else(|| format!("{prefix}KernelError")),
        RoleApiRole::Verifier => find_public_item(source, "pub enum ", "Error")
            .unwrap_or_else(|| format!("Verify{prefix}Error")),
    }
}
