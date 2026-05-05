use super::super::super::role::RoleApiRole;
use super::super::super::{CommitmentRustApi, RoleApiProgramSource, StageRustApi};

const ROLE: RoleApiRole = RoleApiRole::Prover;

pub(super) fn push_commitment_execution(
    source: &mut String,
    commitment: &CommitmentRustApi,
    instrumentation_prefix: Option<&str>,
) {
    let prover_fn = commitment
        .prover_entrypoint()
        .unwrap_or("missing_commitment_prover_function");
    let program_arg = commitment.program_argument_prefix(ROLE);
    if let Some(prefix) = instrumentation_prefix {
        source.push_str(&format!(
            "    let _{field}_span = tracing::info_span!(\"{prefix}.{field}\").entered();\n",
            field = commitment.field_name
        ));
    }
    source.push_str(&format!(
        "    let {field} = {module}::{prover_fn}(\n        {program_arg}inputs.commitment_inputs,\n        inputs.prover_setup,\n        transcript,\n    )?;\n",
        field = commitment.field_name,
        module = commitment.module_alias
    ));
    if instrumentation_prefix.is_some() {
        source.push_str(&format!("    drop(_{}_span);\n", commitment.field_name));
    }
}

pub(super) fn push_stage_execution(
    source: &mut String,
    stage: &StageRustApi,
    instrumentation_prefix: Option<&str>,
) {
    let prover_fn = stage
        .prover_entrypoint()
        .unwrap_or("missing_prover_function");
    let program_arg = stage.program_argument_prefix(ROLE);
    if let Some(prefix) = instrumentation_prefix {
        source.push_str(&format!(
            "    let _{field}_span = tracing::info_span!(\"{prefix}.{span}\").entered();\n",
            field = stage.field_name,
            span = generated_stage_span_name(&stage.field_name)
        ));
    }
    source.push_str(&format!(
        "    let {} = {}::{}({program_arg}inputs.{}_executor, transcript)?;\n",
        stage.field_name, stage.module_alias, prover_fn, stage.field_name
    ));
    if instrumentation_prefix.is_some() {
        source.push_str(&format!("    drop(_{}_span);\n", stage.field_name));
    }
}

fn generated_stage_span_name(field_name: &str) -> &str {
    field_name.strip_suffix("_outer").unwrap_or(field_name)
}
