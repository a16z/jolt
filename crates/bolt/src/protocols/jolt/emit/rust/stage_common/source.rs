use crate::emit::rust::EmitError;
use crate::ir::Role;

use super::super::source::{role_filename, role_module_source};

pub(in crate::protocols::jolt::emit::rust) fn stage_role_filename(
    role: &Role,
    prover_filename: &'static str,
    verifier_filename: &'static str,
) -> &'static str {
    role_filename(role, prover_filename, verifier_filename)
}

pub(in crate::protocols::jolt::emit::rust) fn stage_role_module_source(
    role: &Role,
    constants: &str,
    entrypoint: &str,
    prover_sections: impl FnOnce() -> (String, String),
    verifier_sections: impl FnOnce() -> (&'static str, String),
) -> String {
    match role {
        Role::Prover => {
            let (imports, types) = prover_sections();
            role_module_source(&imports, &types, constants, entrypoint)
        }
        Role::Verifier => {
            let (imports, types) = verifier_sections();
            role_module_source(imports, &types, constants, entrypoint)
        }
    }
}

pub(in crate::protocols::jolt::emit::rust) fn stage_fallible_role_module_source(
    role: &Role,
    prover_constants: impl FnOnce() -> Result<String, EmitError>,
    verifier_constants: impl FnOnce() -> Result<String, EmitError>,
    prover_sections: impl FnOnce() -> (String, String, &'static str),
    verifier_sections: impl FnOnce() -> (&'static str, String, &'static str),
) -> Result<String, EmitError> {
    match role {
        Role::Prover => {
            let constants = prover_constants()?;
            let (imports, types, entrypoint) = prover_sections();
            Ok(role_module_source(&imports, &types, &constants, entrypoint))
        }
        Role::Verifier => {
            let constants = verifier_constants()?;
            let (imports, types, entrypoint) = verifier_sections();
            Ok(role_module_source(imports, &types, &constants, entrypoint))
        }
    }
}
