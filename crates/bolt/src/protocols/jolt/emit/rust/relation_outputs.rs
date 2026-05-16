use crate::emit::rust::{push_format, EmitError};
use crate::ir::Role;
use crate::protocols::jolt::verifier_relation_outputs::{
    RelationOutputPlan, StructuredPolynomialEvalPlan, StructuredPolynomialEvalRefPlan,
    StructuredPolynomialKind, StructuredPolynomialPointLength, StructuredPolynomialPointOrder,
    StructuredPolynomialPointPlan, StructuredPolynomialPointSegment,
};

pub fn emit_verifier_relation_output_constants(
    stage_type: &str,
    role: &Role,
    relation_output_values: &[StructuredPolynomialEvalPlan],
    relation_outputs: &[RelationOutputPlan],
) -> Result<String, EmitError> {
    let mut source = String::new();
    emit_relation_output_value_constants(&mut source, stage_type, relation_output_values)?;
    let mut claims = Vec::new();
    for (index, claim) in relation_outputs.iter().enumerate() {
        if let Some(family) = claim.product_families.first() {
            return Err(EmitError::new(format!(
                "{stage_type} relation output product family @{} must be lowered before Rust verifier emission",
                family.symbol
            )));
        }
        let values_name = format!(
            "{}_RELATION_OUTPUT_{index}_STRUCTURED_POLYNOMIAL_EVALS",
            stage_type.to_ascii_uppercase()
        );
        let values = emit_structured_polynomial_eval_refs_slice_or_inline(
            &mut source,
            &values_name,
            &claim.structured_polynomial_evals,
        );
        let eval_families = emit_eval_family_constants(&mut source, stage_type, index, claim);
        let local_scalars = emit_local_scalar_constants(&mut source, stage_type, index, claim);
        claims.push(format!(
            "    {stage_type}RelationOutputPlan {{ relation: {}, structured_polynomial_evals: {values}, eval_families: {eval_families}, local_scalars: {local_scalars}, expected_output: {} }},",
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

fn emit_relation_output_value_constants(
    source: &mut String,
    stage_type: &str,
    relation_output_values: &[StructuredPolynomialEvalPlan],
) -> Result<(), EmitError> {
    let values_name = format!("{}_RELATION_OUTPUT_VALUES", stage_type.to_ascii_uppercase());
    let values = relation_output_values
        .iter()
        .map(|value| {
            Ok(format!(
                "    {stage_type}StructuredPolynomialEvalPlan {{ symbol: {}, polynomial: {}, x_point: {}, y_point: {} }},",
                rust_str(&value.symbol),
                structured_polynomial_kind_expr(stage_type, value.polynomial),
                structured_polynomial_point_expr(stage_type, &value.x_point),
                structured_polynomial_point_expr(stage_type, &value.y_point),
            ))
        })
        .collect::<Result<Vec<_>, EmitError>>()?
        .join("\n");
    push_format(
        source,
        format_args!(
            "pub const {values_name}: &[{stage_type}StructuredPolynomialEvalPlan] = &[\n{values}\n];\n\n"
        ),
    );
    Ok(())
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
    let scalars = rust_str_array(&claim.local_scalars);
    push_format(
        source,
        format_args!("pub const {name}: &[&str] = &[{scalars}];\n"),
    );
    name
}

fn emit_eval_family_constants(
    source: &mut String,
    stage_type: &str,
    claim_index: usize,
    claim: &RelationOutputPlan,
) -> String {
    if claim.eval_families.is_empty() {
        return "&[]".to_owned();
    }
    let upper_stage = stage_type.to_ascii_uppercase();
    let mut family_rows = Vec::new();
    for (family_index, family) in claim.eval_families.iter().enumerate() {
        let prefix = format!("{upper_stage}_RELATION_OUTPUT_{claim_index}_FAMILY_{family_index}");
        let evals_name = format!("{prefix}_EVALS");
        let evals = rust_str_array(&family.evals);
        push_format(
            source,
            format_args!("pub const {evals_name}: &[&str] = &[{evals}];\n"),
        );
        let value_offsets_name = format!("{prefix}_VALUE_TERM_OFFSETS");
        let value_offsets =
            emit_usize_slice_or_inline(source, &value_offsets_name, &family.value_term_offsets);
        let shared_terms_name = format!("{prefix}_SHARED_TERMS");
        let shared_terms = family
            .shared_terms
            .iter()
            .map(|term| {
                format!(
                    "    bolt_verifier_runtime::RelationOutputEvalFamilySharedTermPlan {{ gamma_power_offset: {}, factor: {} }},",
                    term.gamma_power_offset,
                    rust_str(&term.factor)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        push_format(
            source,
            format_args!(
                "pub const {shared_terms_name}: &[bolt_verifier_runtime::RelationOutputEvalFamilySharedTermPlan] = &[\n{shared_terms}\n];\n"
            ),
        );

        let mut item_rows = Vec::new();
        for (term_index, term) in family.item_terms.iter().enumerate() {
            let factors_name = format!("{prefix}_ITEM_TERM_{term_index}_FACTORS");
            let factors = emit_str_slice_or_inline(source, &factors_name, &term.factors);
            item_rows.push(format!(
                "    bolt_verifier_runtime::RelationOutputEvalFamilyItemTermPlan {{ gamma_power_offset: {}, factors: {factors} }},",
                term.gamma_power_offset
            ));
        }
        let item_terms_name = format!("{prefix}_ITEM_TERMS");
        let item_terms = item_rows.join("\n");
        push_format(
            source,
            format_args!(
                "pub const {item_terms_name}: &[bolt_verifier_runtime::RelationOutputEvalFamilyItemTermPlan] = &[\n{item_terms}\n];\n"
            ),
        );
        family_rows.push(format!(
            "    bolt_verifier_runtime::RelationOutputEvalFamilyPlan {{ symbol: {}, gamma: {}, evals: {evals_name}, power_stride: {}, value_term_offsets: {value_offsets}, shared_terms: {shared_terms_name}, item_terms: {item_terms_name} }},",
            rust_str(&family.symbol),
            rust_str(&family.gamma),
            family.power_stride
        ));
    }
    let families_name = format!("{upper_stage}_RELATION_OUTPUT_{claim_index}_FAMILIES");
    let families = family_rows.join("\n");
    push_format(
        source,
        format_args!(
            "pub const {families_name}: &[bolt_verifier_runtime::RelationOutputEvalFamilyPlan] = &[\n{families}\n];\n\n"
        ),
    );
    families_name
}

fn emit_structured_polynomial_eval_refs_slice_or_inline(
    source: &mut String,
    name: &str,
    values: &[StructuredPolynomialEvalRefPlan],
) -> String {
    let rows = values
        .iter()
        .map(structured_polynomial_eval_ref_expr)
        .collect::<Vec<_>>();
    if values.len() <= 4 {
        return format!("&[{}]", rows.join(", "));
    }
    let rows = rows
        .into_iter()
        .map(|row| format!("    {row},"))
        .collect::<Vec<_>>()
        .join("\n");
    push_format(
        source,
        format_args!(
            "pub const {name}: &[bolt_verifier_runtime::StructuredPolynomialEvalRef] = &[\n{rows}\n];\n"
        ),
    );
    name.to_owned()
}

fn structured_polynomial_eval_ref_expr(value: &StructuredPolynomialEvalRefPlan) -> String {
    format!(
        "bolt_verifier_runtime::StructuredPolynomialEvalRef {{ symbol: {}, index: {} }}",
        rust_str(&value.symbol),
        value.index
    )
}

fn emit_str_slice_or_inline(source: &mut String, name: &str, values: &[String]) -> String {
    if values.len() <= 4 {
        return format!("&[{}]", rust_str_array(values));
    }
    let values = rust_str_array(values);
    push_format(
        source,
        format_args!("pub const {name}: &[&str] = &[{values}];\n"),
    );
    name.to_owned()
}

fn emit_usize_slice_or_inline(source: &mut String, name: &str, values: &[usize]) -> String {
    if values.len() <= 4 {
        return format!("&[{}]", usize_array(values));
    }
    let values = usize_array(values);
    push_format(
        source,
        format_args!("pub const {name}: &[usize] = &[{values}];\n"),
    );
    name.to_owned()
}

fn structured_polynomial_kind_expr(
    stage_type: &str,
    polynomial: StructuredPolynomialKind,
) -> String {
    let variant = match polynomial {
        StructuredPolynomialKind::Eq => "Eq",
        StructuredPolynomialKind::EqPlusOne => "EqPlusOne",
        StructuredPolynomialKind::Lt => "Lt",
    };
    format!("{stage_type}StructuredPolynomialKind::{variant}")
}

fn structured_polynomial_point_expr(
    stage_type: &str,
    point: &StructuredPolynomialPointPlan,
) -> String {
    format!(
        "{stage_type}StructuredPolynomialPointPlan {{ source: {}, segment: {}, length: {}, order: {} }}",
        rust_str(&point.source),
        structured_polynomial_point_segment_expr(stage_type, point.segment),
        structured_polynomial_point_length_expr(stage_type, point.length),
        structured_polynomial_point_order_expr(stage_type, point.order),
    )
}

fn structured_polynomial_point_segment_expr(
    stage_type: &str,
    segment: StructuredPolynomialPointSegment,
) -> String {
    let variant = match segment {
        StructuredPolynomialPointSegment::Full => "Full",
        StructuredPolynomialPointSegment::Prefix => "Prefix",
        StructuredPolynomialPointSegment::Suffix => "Suffix",
    };
    format!("{stage_type}StructuredPolynomialPointSegment::{variant}")
}

fn structured_polynomial_point_length_expr(
    stage_type: &str,
    length: StructuredPolynomialPointLength,
) -> String {
    let variant = match length {
        StructuredPolynomialPointLength::Full => "Full",
        StructuredPolynomialPointLength::XPoint => "XPoint",
        StructuredPolynomialPointLength::YPoint => "YPoint",
    };
    format!("{stage_type}StructuredPolynomialPointLength::{variant}")
}

fn structured_polynomial_point_order_expr(
    stage_type: &str,
    order: StructuredPolynomialPointOrder,
) -> String {
    let variant = match order {
        StructuredPolynomialPointOrder::AsIs => "AsIs",
        StructuredPolynomialPointOrder::Reverse => "Reverse",
    };
    format!("{stage_type}StructuredPolynomialPointOrder::{variant}")
}

fn rust_str(value: &str) -> String {
    format!("{value:?}")
}

fn rust_str_array(values: &[String]) -> String {
    values
        .iter()
        .map(|value| rust_str(value))
        .collect::<Vec<_>>()
        .join(", ")
}

fn usize_array(values: &[usize]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}
