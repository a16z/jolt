use crate::ir::Role;

pub(in crate::protocols::jolt::emit::rust) fn role_filename(
    role: &Role,
    prover_filename: &'static str,
    verifier_filename: &'static str,
) -> &'static str {
    match role {
        Role::Prover => prover_filename,
        Role::Verifier => verifier_filename,
    }
}

pub(in crate::protocols::jolt::emit::rust) fn role_module_source(
    imports: &str,
    types: &str,
    constants: &str,
    entrypoint: &str,
) -> String {
    let mut source = String::new();
    source.push_str("#![allow(dead_code)]\n\n");
    source.push_str(imports);
    source.push_str("\n\n");
    source.push_str(types);
    source.push('\n');
    source.push_str(constants);
    source.push('\n');
    source.push_str(entrypoint);
    source
}
