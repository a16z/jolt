use super::super::super::role::RoleApiRole;
use super::super::super::{CommitmentRustApi, RoleApiProgramSource, StageRustApi};

const ROLE: RoleApiRole = RoleApiRole::Verifier;

pub(super) fn push_commitment_execution(source: &mut String, commitment: &CommitmentRustApi) {
    let verifier_fn = commitment
        .verifier_entrypoint()
        .unwrap_or("missing_commitment_verifier_function");
    let program_arg = commitment.program_argument_prefix(ROLE);
    source.push_str(&format!(
        "    let {field} = {module}::{verifier_fn}({program_arg}&proof.commitments, transcript)?;\n",
        field = commitment.field_name,
        module = commitment.module_alias,
    ));
}

pub(super) fn push_stage_execution(source: &mut String, stage: &StageRustApi) {
    let verifier_fn = stage
        .verifier_entrypoint()
        .unwrap_or("missing_verifier_function");
    let mut args = vec![format!("&proof.{}", stage.field_name)];
    if let Some(program_arg) = stage.program_argument(ROLE) {
        args.insert(0, program_arg);
    }
    for input in stage.verifier_inputs() {
        args.push(format!(
            "inputs.{}_{}",
            stage.field_name,
            input.kind.field_suffix()
        ));
    }
    args.push("transcript".to_owned());
    source.push_str(&format!(
        "    let {} = {}::{}({})?;\n",
        stage.field_name,
        stage.module_alias,
        verifier_fn,
        args.join(", ")
    ));
}
