use crate::emit::rust::{push_format, EmitError};
use crate::ir::Role;
use crate::protocols::jolt::verifier_relation_outputs::RelationOutputPlan;

pub fn emit_verifier_relation_output_constants(
    stage_type: &str,
    role: &Role,
    relation_outputs: &[RelationOutputPlan],
) -> Result<String, EmitError> {
    let mut source = String::new();
    let mut claims = Vec::new();
    for (index, claim) in relation_outputs.iter().enumerate() {
        let local_scalars = emit_local_scalar_constants(&mut source, stage_type, index, claim);
        claims.push(format!(
            "    {stage_type}RelationOutputPlan {{ relation: {}, local_scalars: {local_scalars}, expected_output: {} }},",
            super::plan_tokens::role_relation_kind_expr(stage_type, role, &claim.relation)?,
            rust_str(&claim.expected_output)
        ));
    }
    let claims = claims.join("\n");
    let claims_name = format!("{}_RELATION_OUTPUTS", stage_type.to_ascii_uppercase());
    push_format(
        &mut source,
        format_args!(
            "pub const {claims_name}: &[{stage_type}RelationOutputPlan] = &[\n{claims}\n];\n\n"
        ),
    );
    Ok(source)
}

fn emit_local_scalar_constants(
    source: &mut String,
    stage_type: &str,
    claim_index: usize,
    claim: &RelationOutputPlan,
) -> String {
    if claim.local_scalars.is_empty() {
        return "&[]".to_owned();
    }
    let name = format!(
        "{}_RELATION_OUTPUT_{claim_index}_LOCAL_SCALARS",
        stage_type.to_ascii_uppercase()
    );
    let scalars = claim
        .local_scalar_symbols()
        .map(|symbol| rust_str(symbol))
        .collect::<Vec<_>>()
        .join(", ");
    push_format(
        source,
        format_args!("pub const {name}: &[&str] = &[{scalars}];\n"),
    );
    name
}

fn rust_str(value: &str) -> String {
    format!("{value:?}")
}
