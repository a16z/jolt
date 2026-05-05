use super::super::super::role::RoleApiRole;
use super::super::super::{CommitmentRustApi, RoleApiProgramSource, StageRustApi};

const ROLE: RoleApiRole = RoleApiRole::Prover;

pub(super) fn push_commitment_execution(source: &mut String, commitment: &CommitmentRustApi) {
    let prover_fn = commitment
        .prover_entrypoint()
        .unwrap_or("missing_commitment_prover_function");
    let program_arg = commitment.program_argument_prefix(ROLE);
    source.push_str(&format!(
        "    let {field} = {module}::{prover_fn}(\n        {program_arg}inputs.commitment_inputs,\n        inputs.prover_setup,\n        transcript,\n    )?;\n",
        field = commitment.field_name,
        module = commitment.module_alias
    ));
}

pub(super) fn push_stage_execution(source: &mut String, stage: &StageRustApi) {
    let prover_fn = stage
        .prover_entrypoint()
        .unwrap_or("missing_prover_function");
    let program_arg = stage.program_argument_prefix(ROLE);
    source.push_str(&format!(
        "    let {} = {}::{}({program_arg}inputs.{}_executor, transcript)?;\n",
        stage.field_name, stage.module_alias, prover_fn, stage.field_name
    ));
}
