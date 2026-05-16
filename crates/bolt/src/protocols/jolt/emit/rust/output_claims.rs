use crate::emit::rust::{push_format, EmitError};
use crate::ir::Role;
use crate::protocols::jolt::verifier_output_claims::{
    StructuredPolynomialKind, StructuredPolynomialPointLength, StructuredPolynomialPointOrder,
    StructuredPolynomialPointPlan, StructuredPolynomialPointSegment, SumcheckOutputClaimPlan,
    SumcheckOutputFunctionKind,
};

pub fn emit_verifier_output_claim_constants(
    stage_type: &str,
    role: &Role,
    output_claims: &[SumcheckOutputClaimPlan],
) -> Result<String, EmitError> {
    let mut source = String::new();
    let mut claims = Vec::new();
    for (index, claim) in output_claims.iter().enumerate() {
        let values_name = format!(
            "{}_SUMCHECK_OUTPUT_CLAIM_{index}_VALUES",
            stage_type.to_ascii_uppercase()
        );
        let values = claim
            .polynomial_evals
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
            &mut source,
            format_args!(
                "pub const {values_name}: &[{stage_type}StructuredPolynomialEvalPlan] = &[\n{values}\n];\n\n"
            ),
        );
        let eval_families = emit_eval_family_constants(&mut source, stage_type, index, claim);
        let product_families = emit_product_family_constants(&mut source, stage_type, index, claim);
        let function_families =
            emit_function_family_constants(&mut source, stage_type, index, claim)?;
        let local_scalars = emit_local_scalar_constants(&mut source, stage_type, index, claim);
        claims.push(format!(
            "    {stage_type}SumcheckOutputClaimPlan {{ relation: {}, polynomial_evals: {values_name}, eval_families: {eval_families}, product_families: {product_families}, function_families: {function_families}, local_scalars: {local_scalars}, expected_output: {} }},",
            super::plan_tokens::role_relation_kind_expr(stage_type, role, &claim.relation)?,
            rust_str(&claim.expected_output)
        ));
    }
    let claims = claims.join("\n");
    let claims_name = format!("{}_SUMCHECK_OUTPUT_CLAIMS", stage_type.to_ascii_uppercase());
    push_format(
        &mut source,
        format_args!(
            "pub const {claims_name}: &[{stage_type}SumcheckOutputClaimPlan] = &[\n{claims}\n];\n\n"
        ),
    );
    Ok(source)
}

fn emit_local_scalar_constants(
    source: &mut String,
    stage_type: &str,
    claim_index: usize,
    claim: &SumcheckOutputClaimPlan,
) -> String {
    if claim.local_scalars.is_empty() {
        return "&[]".to_owned();
    }
    let name = format!(
        "{}_SUMCHECK_OUTPUT_CLAIM_{claim_index}_LOCAL_SCALARS",
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
    claim: &SumcheckOutputClaimPlan,
) -> String {
    if claim.eval_families.is_empty() {
        return "&[]".to_owned();
    }
    let upper_stage = stage_type.to_ascii_uppercase();
    let mut family_rows = Vec::new();
    for (family_index, family) in claim.eval_families.iter().enumerate() {
        let prefix =
            format!("{upper_stage}_SUMCHECK_OUTPUT_CLAIM_{claim_index}_FAMILY_{family_index}");
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
                    "    bolt_verifier_runtime::SumcheckOutputEvalFamilySharedTermPlan {{ gamma_power_offset: {}, factor: {} }},",
                    term.gamma_power_offset,
                    rust_str(&term.factor)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        push_format(
            source,
            format_args!(
                "pub const {shared_terms_name}: &[bolt_verifier_runtime::SumcheckOutputEvalFamilySharedTermPlan] = &[\n{shared_terms}\n];\n"
            ),
        );

        let mut item_rows = Vec::new();
        for (term_index, term) in family.item_terms.iter().enumerate() {
            let factors_name = format!("{prefix}_ITEM_TERM_{term_index}_FACTORS");
            let factors = emit_str_slice_or_inline(source, &factors_name, &term.factors);
            item_rows.push(format!(
                "    bolt_verifier_runtime::SumcheckOutputEvalFamilyItemTermPlan {{ gamma_power_offset: {}, factors: {factors} }},",
                term.gamma_power_offset
            ));
        }
        let item_terms_name = format!("{prefix}_ITEM_TERMS");
        let item_terms = item_rows.join("\n");
        push_format(
            source,
            format_args!(
                "pub const {item_terms_name}: &[bolt_verifier_runtime::SumcheckOutputEvalFamilyItemTermPlan] = &[\n{item_terms}\n];\n"
            ),
        );
        family_rows.push(format!(
            "    bolt_verifier_runtime::SumcheckOutputEvalFamilyPlan {{ symbol: {}, gamma: {}, evals: {evals_name}, power_stride: {}, value_term_offsets: {value_offsets}, shared_terms: {shared_terms_name}, item_terms: {item_terms_name} }},",
            rust_str(&family.symbol),
            rust_str(&family.gamma),
            family.power_stride
        ));
    }
    let families_name = format!("{upper_stage}_SUMCHECK_OUTPUT_CLAIM_{claim_index}_FAMILIES");
    let families = family_rows.join("\n");
    push_format(
        source,
        format_args!(
            "pub const {families_name}: &[bolt_verifier_runtime::SumcheckOutputEvalFamilyPlan] = &[\n{families}\n];\n\n"
        ),
    );
    families_name
}

fn emit_product_family_constants(
    source: &mut String,
    stage_type: &str,
    claim_index: usize,
    claim: &SumcheckOutputClaimPlan,
) -> String {
    if claim.product_families.is_empty() {
        return "&[]".to_owned();
    }
    let upper_stage = stage_type.to_ascii_uppercase();
    let mut family_rows = Vec::new();
    for (family_index, family) in claim.product_families.iter().enumerate() {
        let prefix = format!(
            "{upper_stage}_SUMCHECK_OUTPUT_CLAIM_{claim_index}_PRODUCT_FAMILY_{family_index}"
        );
        let mut term_rows = Vec::new();
        for (term_index, term) in family.terms.iter().enumerate() {
            let evals_name = format!("{prefix}_TERM_{term_index}_EVALS");
            let evals = emit_str_slice_or_inline(source, &evals_name, &term.evals);
            let eval_families_name = format!("{prefix}_TERM_{term_index}_EVAL_FAMILIES");
            let eval_families =
                emit_str_slice_or_inline(source, &eval_families_name, &term.eval_families);
            let factors_name = format!("{prefix}_TERM_{term_index}_FACTORS");
            let factors = emit_str_slice_or_inline(source, &factors_name, &term.factors);
            term_rows.push(format!(
                "    bolt_verifier_runtime::SumcheckOutputProductFamilyTermPlan {{ gamma_power_offset: {}, evals: {evals}, eval_families: {eval_families}, factors: {factors} }},",
                term.gamma_power_offset
            ));
        }
        let terms_name = format!("{prefix}_TERMS");
        let terms = term_rows.join("\n");
        push_format(
            source,
            format_args!(
                "pub const {terms_name}: &[bolt_verifier_runtime::SumcheckOutputProductFamilyTermPlan] = &[\n{terms}\n];\n"
            ),
        );
        family_rows.push(format!(
            "    bolt_verifier_runtime::SumcheckOutputProductFamilyPlan {{ symbol: {}, gamma: {}, terms: {terms_name} }},",
            rust_str(&family.symbol),
            optional_rust_str(family.gamma.as_deref()),
        ));
    }
    let families_name =
        format!("{upper_stage}_SUMCHECK_OUTPUT_CLAIM_{claim_index}_PRODUCT_FAMILIES");
    let families = family_rows.join("\n");
    push_format(
        source,
        format_args!(
            "pub const {families_name}: &[bolt_verifier_runtime::SumcheckOutputProductFamilyPlan] = &[\n{families}\n];\n\n"
        ),
    );
    families_name
}

fn emit_function_family_constants(
    source: &mut String,
    stage_type: &str,
    claim_index: usize,
    claim: &SumcheckOutputClaimPlan,
) -> Result<String, EmitError> {
    if claim.function_families.is_empty() {
        return Ok("&[]".to_owned());
    }
    let upper_stage = stage_type.to_ascii_uppercase();
    let mut family_rows = Vec::new();
    for (family_index, family) in claim.function_families.iter().enumerate() {
        let prefix = format!(
            "{upper_stage}_SUMCHECK_OUTPUT_CLAIM_{claim_index}_FUNCTION_FAMILY_{family_index}"
        );
        let mut term_rows = Vec::new();
        for (term_index, term) in family.terms.iter().enumerate() {
            let factors_name = format!("{prefix}_TERM_{term_index}_FACTORS");
            let factors = emit_str_slice_or_inline(source, &factors_name, &term.factors);
            term_rows.push(format!(
                "    bolt_verifier_runtime::SumcheckOutputFunctionFamilyTermPlan {{ gamma_power_offset: {}, function: {}, eval: {}, factors: {factors} }},",
                term.gamma_power_offset,
                output_function_kind_expr(term.function),
                rust_str(&term.eval)
            ));
        }
        let terms_name = format!("{prefix}_TERMS");
        let terms = term_rows.join("\n");
        push_format(
            source,
            format_args!(
                "pub const {terms_name}: &[bolt_verifier_runtime::SumcheckOutputFunctionFamilyTermPlan] = &[\n{terms}\n];\n"
            ),
        );
        family_rows.push(format!(
            "    bolt_verifier_runtime::SumcheckOutputFunctionFamilyPlan {{ symbol: {}, gamma: {}, terms: {terms_name} }},",
            rust_str(&family.symbol),
            optional_rust_str(family.gamma.as_deref()),
        ));
    }
    let families_name =
        format!("{upper_stage}_SUMCHECK_OUTPUT_CLAIM_{claim_index}_FUNCTION_FAMILIES");
    let families = family_rows.join("\n");
    push_format(
        source,
        format_args!(
            "pub const {families_name}: &[bolt_verifier_runtime::SumcheckOutputFunctionFamilyPlan] = &[\n{families}\n];\n\n"
        ),
    );
    Ok(families_name)
}

fn output_function_kind_expr(function: SumcheckOutputFunctionKind) -> &'static str {
    match function {
        SumcheckOutputFunctionKind::BooleanZero => {
            "bolt_verifier_runtime::SumcheckOutputFunctionKind::BooleanZero"
        }
    }
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

fn optional_rust_str(value: Option<&str>) -> String {
    value.map_or_else(
        || "None".to_owned(),
        |value| format!("Some({})", rust_str(value)),
    )
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
