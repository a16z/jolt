use crate::ir::Role;

use super::super::super::types::ProtocolArtifactExtension;
use super::super::names::RoleApiNames;
use super::super::role::RoleApiRole;

pub(super) fn role_module_source(
    role: &Role,
    extension: Option<&ProtocolArtifactExtension>,
    protocol_snake: &str,
    names: &RoleApiNames,
) -> String {
    let role = RoleApiRole::from_role(role);
    if let Some(extension) = extension {
        return role.extension_lib_module(extension).to_owned();
    }
    match role {
        RoleApiRole::Prover => prover_lib_module_source(protocol_snake, names),
        RoleApiRole::Verifier => verifier_lib_module_source(protocol_snake, names),
    }
}

fn prover_lib_module_source(protocol_snake: &str, names: &RoleApiNames) -> String {
    let role_names = names.role(RoleApiRole::Prover);
    format!(
        "#[rustfmt::skip]\npub mod prover;\npub mod stages;\n\npub use prover::{{\n    default_prover_programs, prove_{protocol_snake}, prove_{protocol_snake}_with_programs,\n    {default_transcript_alias}, {prove_error}, {prover_artifacts}, {prover_inputs},\n    {prover_programs},\n}};",
        default_transcript_alias = names.default_transcript_alias,
        prove_error = role_names.error,
        prover_artifacts = role_names.artifacts,
        prover_inputs = role_names.inputs,
        prover_programs = role_names.programs,
    )
}

fn verifier_lib_module_source(protocol_snake: &str, names: &RoleApiNames) -> String {
    let role_names = names.role(RoleApiRole::Verifier);
    format!(
        "pub mod stages;\n#[rustfmt::skip]\npub mod verifier;\n\npub use verifier::{{\n    default_verifier_programs, verify_{protocol_snake}, verify_{protocol_snake}_with_programs, {named_eval}, {proof},\n    {stage_proof}, {sumcheck_output}, {verification_artifacts}, {verifier_inputs},\n    {verifier_programs}, {verify_error},\n}};",
        named_eval = names.named_eval,
        proof = names.proof,
        stage_proof = names.stage_proof,
        sumcheck_output = names.sumcheck_output,
        verification_artifacts = role_names.artifacts,
        verifier_inputs = role_names.inputs,
        verifier_programs = role_names.programs,
        verify_error = role_names.error,
    )
}
