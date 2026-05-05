use super::super::names::RoleApiNames;

pub(super) fn push_type_aliases(
    source: &mut String,
    names: &RoleApiNames,
    field_type: &str,
    runtime_named_eval_type: &str,
    runtime_sumcheck_output_type: &str,
    runtime_stage_proof_type: &str,
) {
    source.push_str(&format!(
        "pub type {} = {runtime_named_eval_type}<{field_type}>;\n\
         pub type {} = {runtime_sumcheck_output_type}<{field_type}>;\n\
         pub type {} = {runtime_stage_proof_type}<{field_type}>;\n\n",
        names.named_eval, names.sumcheck_output, names.stage_proof,
    ));
}
