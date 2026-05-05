use super::super::source_scan::{find_public_const_of_type, find_type_with_suffix};
use super::super::RoleApiProgramBinding;
use super::entrypoints::discover_entrypoints;

pub(super) struct DiscoveredProgram {
    pub(super) program_type: Option<String>,
    pub(super) program_const: Option<String>,
}

pub(super) fn discover_program_binding(
    source: &str,
    type_suffix: &str,
    prover_prefixes: &[&str],
) -> RoleApiProgramBinding {
    let program = discover_program(source, type_suffix);
    let entrypoints = discover_entrypoints(source, prover_prefixes);
    RoleApiProgramBinding {
        verifier_fn: entrypoints.verifier_fn,
        with_program_verifier_fn: entrypoints.with_program_verifier_fn,
        program_type: program.program_type,
        program_const: program.program_const,
        prover_fn: entrypoints.prover_fn,
        with_program_prover_fn: entrypoints.with_program_prover_fn,
    }
}

pub(super) fn discover_program(source: &str, type_suffix: &str) -> DiscoveredProgram {
    let program_type = find_type_with_suffix(source, type_suffix);
    let program_const = program_type
        .as_deref()
        .and_then(|program_type| find_public_const_of_type(source, program_type));
    DiscoveredProgram {
        program_type,
        program_const,
    }
}
