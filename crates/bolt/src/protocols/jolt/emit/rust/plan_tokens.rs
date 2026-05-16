use std::collections::BTreeSet;

use crate::emit::rust::EmitError;
use crate::ir::Role;
use crate::protocols::jolt::rust_target_plan::{
    ClaimKind, FieldExprKind, JoltVerifierRelationKind, OpeningEqualityMode, PcsProofMode,
    ProgramStepKind, RustTargetPlanError, ScalarExprKind, SumcheckPointOrder,
    TranscriptSqueezeKind,
};
use crate::protocols::jolt::verifier_plan::{
    VerifierFieldExprPlan, VerifierScalarExprOperand, VerifierScalarExprPlan,
};
use crate::protocols::jolt::verifier_value_rows::{CpuFieldExprPlan, CpuScalarExprPlan};
use crate::protocols::jolt::verifier_values::{
    VerifierFieldVectorValueRef, VerifierFieldVectorValueSet, VerifierPointSourceSet,
    VerifierScalarSourceSet, VerifierScalarValueSet,
};

pub(super) fn role_program_step_kind_expr(
    stage_type_prefix: &str,
    role: &Role,
    kind: &str,
) -> Result<String, EmitError> {
    if role == &Role::Prover {
        return Ok(format!("{kind:?}"));
    }
    program_step_kind_expr(stage_type_prefix, kind)
}

pub(super) fn role_transcript_squeeze_kind_expr(
    stage_type_prefix: &str,
    role: &Role,
    kind: &str,
) -> Result<String, EmitError> {
    if role == &Role::Prover {
        return Ok(format!("{kind:?}"));
    }
    transcript_squeeze_kind_expr(stage_type_prefix, kind)
}

pub(super) fn role_claim_kind_expr(
    stage_type_prefix: &str,
    role: &Role,
    kind: &str,
) -> Result<String, EmitError> {
    if role == &Role::Prover {
        return Ok(format!("{kind:?}"));
    }
    claim_kind_expr(stage_type_prefix, kind)
}

pub(super) fn role_field_expr_kind_expr(
    stage_type_prefix: &str,
    role: &Role,
    formula: &str,
) -> Result<String, EmitError> {
    if role == &Role::Prover {
        return Ok(format!("{formula:?}"));
    }
    field_expr_kind_expr(stage_type_prefix, formula)
}

pub(super) fn rust_str_slice_expr(values: &[String]) -> String {
    if values.is_empty() {
        return "&[]".to_owned();
    }
    let values = values
        .iter()
        .map(|value| format!("{value:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("&[{values}]")
}

pub(super) fn rust_str(value: &str) -> String {
    format!("{value:?}")
}

pub(super) fn rust_option_str(value: Option<&str>) -> String {
    value.map_or_else(
        || "None".to_owned(),
        |value| format!("Some({})", rust_str(value)),
    )
}

pub(super) fn emit_str_array(name: &str, values: &[String]) -> String {
    if values.is_empty() {
        return format!("pub const {name}: &[&str] = &[];\n\n");
    }
    if let [value] = values {
        return format!("pub const {name}: &[&str] = &[{}];\n\n", rust_str(value));
    }
    let entries = values
        .iter()
        .map(|value| format!("    {},", rust_str(value)))
        .collect::<Vec<_>>()
        .join("\n");
    format!("pub const {name}: &[&str] = &[\n{entries}\n];\n\n")
}

pub(super) fn emit_usize_array(name: &str, values: &[usize]) -> String {
    let entries = values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("pub const {name}: &[usize] = &[{entries}];\n\n")
}

pub(super) fn intern_str_array(
    source: &mut String,
    arrays: &mut Vec<(Vec<String>, String)>,
    name_prefix: &str,
    values: &[String],
) -> String {
    if let Some((_, name)) = arrays
        .iter()
        .find(|(existing, _)| existing.as_slice() == values)
    {
        return name.clone();
    }
    let name = format!("{name_prefix}_{}", arrays.len());
    source.push_str(&emit_str_array(&name, values));
    arrays.push((values.to_vec(), name.clone()));
    name
}

pub(super) fn require_supported_symbol(
    kind: &str,
    actual: &str,
    expected: &str,
) -> Result<(), EmitError> {
    if actual == expected {
        Ok(())
    } else {
        Err(EmitError::new(format!(
            "unsupported {kind} @{actual}; expected @{expected}"
        )))
    }
}

pub(super) fn verify_count(
    kind: &str,
    symbol: &str,
    expected: usize,
    actual: usize,
) -> Result<(), EmitError> {
    if expected == actual {
        Ok(())
    } else {
        Err(EmitError::new(format!(
            "{kind} @{symbol} count mismatch: expected {expected}, got {actual}"
        )))
    }
}

pub(super) fn symbols<'a>(values: impl Iterator<Item = &'a String>) -> BTreeSet<String> {
    values.cloned().collect()
}

pub(super) struct FieldExprFlowVerification<'a> {
    pub cpu_exprs: &'a [CpuFieldExprPlan],
    pub verifier_exprs: Option<&'a [VerifierFieldExprPlan]>,
    pub field_values: &'a VerifierScalarSourceSet,
    pub verifier_field_values: Option<&'a VerifierScalarValueSet>,
}

pub(super) fn verify_field_expr_flow(
    verification: FieldExprFlowVerification<'_>,
) -> Result<(), EmitError> {
    let FieldExprFlowVerification {
        cpu_exprs,
        verifier_exprs,
        field_values,
        verifier_field_values,
    } = verification;
    if let (Some(exprs), Some(field_values)) = (verifier_exprs, verifier_field_values) {
        for expr in exprs {
            for operand in &expr.operands {
                if !field_values.contains_ref(operand) {
                    return Err(EmitError::new(format!(
                        "field expr @{} references missing field value @{}",
                        expr.symbol,
                        operand.symbol()
                    )));
                }
            }
        }
    } else {
        for expr in cpu_exprs {
            verify_count(
                "field expr operands",
                &expr.symbol,
                expr.operand_names.len(),
                expr.operands.len(),
            )?;
            let _kind = FieldExprKind::from_cpu_attr(&expr.formula)
                .map_err(|error| EmitError::new(error.to_string()))?;
            for operand in &expr.operands {
                if !field_values.contains(operand) {
                    return Err(EmitError::new(format!(
                        "field expr @{} references missing field value @{operand}",
                        expr.symbol
                    )));
                }
            }
        }
    }
    Ok(())
}

pub(super) struct ScalarExprVerification<'a> {
    pub stage: &'static str,
    pub symbol: &'a str,
    pub formula: &'a str,
    pub operand_names: &'a [String],
    pub operands: &'a [String],
    pub field_values: &'a VerifierScalarSourceSet,
    pub field_vector_values: Option<&'a VerifierFieldVectorValueSet>,
    pub point_values: Option<&'a VerifierPointSourceSet>,
}

pub(super) fn verify_scalar_expr_operands(
    verification: ScalarExprVerification<'_>,
) -> Result<(), EmitError> {
    let ScalarExprVerification {
        stage,
        symbol,
        formula,
        operand_names,
        operands,
        field_values,
        field_vector_values,
        point_values,
    } = verification;
    verify_count(
        "scalar expr operands",
        symbol,
        operand_names.len(),
        operands.len(),
    )?;
    let kind = ScalarExprKind::from_cpu_attr(formula)
        .map_err(|error| EmitError::new(error.to_string()))?;
    match kind {
        ScalarExprKind::FieldVectorSum | ScalarExprKind::FieldVectorProduct => {
            let Some(field_vector_values) = field_vector_values else {
                return Err(EmitError::new(format!(
                    "{stage} scalar expr @{symbol} uses unsupported field-vector formula `{formula}`"
                )));
            };
            verify_count("field vector expr operands", symbol, 1, operands.len())?;
            let operand = &operands[0];
            if !field_vector_values
                .contains_ref(&VerifierFieldVectorValueRef::new(operand.as_str()))
            {
                return Err(EmitError::new(format!(
                    "field vector expr @{symbol} references missing field vector @{operand}"
                )));
            }
        }
        ScalarExprKind::PowerStridedWeightedSum { .. } => {
            for operand in operands {
                if !field_values.contains(operand) {
                    return Err(EmitError::new(format!(
                        "scalar expr @{symbol} references missing field value @{operand}"
                    )));
                }
            }
        }
        ScalarExprKind::StructuredPolynomial { .. } => {
            verify_count(
                "structured polynomial scalar expr operands",
                symbol,
                2,
                operands.len(),
            )?;
            for operand in operands {
                if !point_values.is_some_and(|values| values.contains(operand)) {
                    return Err(EmitError::new(format!(
                        "structured polynomial scalar expr @{symbol} references missing point value @{operand}"
                    )));
                }
            }
        }
        ScalarExprKind::PointElement { .. } => {
            verify_count(
                "point element scalar expr operands",
                symbol,
                1,
                operands.len(),
            )?;
            let Some(point_values) = point_values else {
                return Err(EmitError::new(format!(
                    "{stage} scalar expr @{symbol} uses point element formula without point sources"
                )));
            };
            let operand = &operands[0];
            if !point_values.contains(operand) {
                return Err(EmitError::new(format!(
                    "point element scalar expr @{symbol} references missing point @{operand}"
                )));
            }
        }
    }
    Ok(())
}

pub(super) struct VerifierScalarExprVerification<'a> {
    pub stage: &'static str,
    pub expr: &'a VerifierScalarExprPlan,
    pub field_values: &'a VerifierScalarValueSet,
    pub field_vector_values: Option<&'a VerifierFieldVectorValueSet>,
    pub point_values: Option<&'a VerifierPointSourceSet>,
}

pub(super) fn verify_verifier_scalar_expr_operands(
    verification: VerifierScalarExprVerification<'_>,
) -> Result<(), EmitError> {
    let VerifierScalarExprVerification {
        stage,
        expr,
        field_values,
        field_vector_values,
        point_values,
    } = verification;
    match expr.kind {
        ScalarExprKind::FieldVectorSum | ScalarExprKind::FieldVectorProduct => {
            let Some(field_vector_values) = field_vector_values else {
                return Err(EmitError::new(format!(
                    "{stage} scalar expr @{} uses field-vector formula without field-vector values",
                    expr.symbol
                )));
            };
            verify_count(
                "field vector expr operands",
                &expr.symbol,
                1,
                expr.operands.len(),
            )?;
            for operand in &expr.operands {
                let VerifierScalarExprOperand::FieldVector(value_ref) = operand else {
                    return Err(EmitError::new(format!(
                        "{stage} scalar expr @{} expected field-vector operand @{}",
                        expr.symbol,
                        operand.symbol()
                    )));
                };
                if !field_vector_values.contains_ref(value_ref) {
                    return Err(EmitError::new(format!(
                        "field vector expr @{} references missing field vector @{}",
                        expr.symbol,
                        value_ref.symbol()
                    )));
                }
            }
        }
        ScalarExprKind::PowerStridedWeightedSum { .. } => {
            for operand in &expr.operands {
                let VerifierScalarExprOperand::Field(value_ref) = operand else {
                    return Err(EmitError::new(format!(
                        "{stage} scalar expr @{} expected field operand @{}",
                        expr.symbol,
                        operand.symbol()
                    )));
                };
                if !field_values.contains_ref(value_ref) {
                    return Err(EmitError::new(format!(
                        "scalar expr @{} references missing field value @{}",
                        expr.symbol,
                        value_ref.symbol()
                    )));
                }
            }
        }
        ScalarExprKind::StructuredPolynomial { .. } => {
            verify_count(
                "structured polynomial scalar expr operands",
                &expr.symbol,
                2,
                expr.operands.len(),
            )?;
            for operand in &expr.operands {
                let VerifierScalarExprOperand::Point(value_ref) = operand else {
                    return Err(EmitError::new(format!(
                        "{stage} scalar expr @{} expected point operand @{}",
                        expr.symbol,
                        operand.symbol()
                    )));
                };
                if !point_values.is_some_and(|values| values.contains_ref(value_ref)) {
                    return Err(EmitError::new(format!(
                        "structured polynomial scalar expr @{} references missing point value @{}",
                        expr.symbol,
                        value_ref.symbol()
                    )));
                }
            }
        }
        ScalarExprKind::PointElement { .. } => {
            verify_count(
                "point element scalar expr operands",
                &expr.symbol,
                1,
                expr.operands.len(),
            )?;
            let Some(point_values) = point_values else {
                return Err(EmitError::new(format!(
                    "{stage} scalar expr @{} uses point element formula without point sources",
                    expr.symbol
                )));
            };
            for operand in &expr.operands {
                let VerifierScalarExprOperand::Point(value_ref) = operand else {
                    return Err(EmitError::new(format!(
                        "{stage} scalar expr @{} expected point operand @{}",
                        expr.symbol,
                        operand.symbol()
                    )));
                };
                if !point_values.contains_ref(value_ref) {
                    return Err(EmitError::new(format!(
                        "point element scalar expr @{} references missing point @{}",
                        expr.symbol,
                        value_ref.symbol()
                    )));
                }
            }
        }
    }
    Ok(())
}

pub(super) struct ScalarExprFlowVerification<'a> {
    pub stage: &'static str,
    pub cpu_exprs: &'a [CpuScalarExprPlan],
    pub verifier_exprs: Option<&'a [VerifierScalarExprPlan]>,
    pub field_values: &'a VerifierScalarSourceSet,
    pub verifier_field_values: Option<&'a VerifierScalarValueSet>,
    pub field_vector_values: Option<&'a VerifierFieldVectorValueSet>,
    pub point_values: Option<&'a VerifierPointSourceSet>,
}

pub(super) fn verify_scalar_expr_flow(
    verification: ScalarExprFlowVerification<'_>,
) -> Result<(), EmitError> {
    let ScalarExprFlowVerification {
        stage,
        cpu_exprs,
        verifier_exprs,
        field_values,
        verifier_field_values,
        field_vector_values,
        point_values,
    } = verification;
    if let (Some(exprs), Some(field_values)) = (verifier_exprs, verifier_field_values) {
        for expr in exprs {
            verify_verifier_scalar_expr_operands(VerifierScalarExprVerification {
                stage,
                expr,
                field_values,
                field_vector_values,
                point_values,
            })?;
        }
    } else {
        for expr in cpu_exprs {
            verify_scalar_expr_operands(ScalarExprVerification {
                stage,
                symbol: &expr.symbol,
                formula: &expr.formula,
                operand_names: &expr.operand_names,
                operands: &expr.operands,
                field_values,
                field_vector_values,
                point_values,
            })?;
        }
    }
    Ok(())
}

pub(super) fn role_relation_kind_expr(
    stage_type_prefix: &str,
    role: &Role,
    relation: &str,
) -> Result<String, EmitError> {
    if role == &Role::Prover {
        return Ok(format!("{relation:?}"));
    }
    relation_kind_expr(stage_type_prefix, relation)
}

pub(super) fn role_sumcheck_point_order_expr(
    role: &Role,
    point_order: &str,
) -> Result<String, EmitError> {
    if role == &Role::Prover {
        return Ok(format!("{point_order:?}"));
    }
    sumcheck_point_order_expr(point_order)
}

pub(super) fn role_optional_relation_kind_expr(
    stage_type_prefix: &str,
    role: &Role,
    relation: Option<&str>,
) -> Result<String, EmitError> {
    relation
        .map(|relation| role_relation_kind_expr(stage_type_prefix, role, relation))
        .transpose()
        .map(|relation| {
            relation.map_or_else(|| "None".to_owned(), |relation| format!("Some({relation})"))
        })
}

pub(super) fn role_opening_equality_mode_expr(
    stage_type_prefix: &str,
    role: &Role,
    mode: &str,
) -> Result<String, EmitError> {
    if role == &Role::Prover {
        return Ok(format!("{mode:?}"));
    }
    opening_equality_mode_expr(stage_type_prefix, mode)
}

fn program_step_kind_expr(stage_type_prefix: &str, kind: &str) -> Result<String, EmitError> {
    let variant = ProgramStepKind::from_cpu_attr(kind)
        .map_err(plan_error)?
        .rust_variant();
    Ok(format!("{stage_type_prefix}ProgramStepKind::{variant}"))
}

fn transcript_squeeze_kind_expr(stage_type_prefix: &str, kind: &str) -> Result<String, EmitError> {
    let variant = TranscriptSqueezeKind::from_cpu_attr(kind)
        .map_err(plan_error)?
        .rust_variant();
    Ok(format!(
        "{stage_type_prefix}TranscriptSqueezeKind::{variant}"
    ))
}

pub(super) fn claim_kind_expr(stage_type_prefix: &str, kind: &str) -> Result<String, EmitError> {
    let variant = ClaimKind::from_cpu_attr(kind)
        .map_err(plan_error)?
        .rust_variant();
    Ok(format!("{stage_type_prefix}ClaimKind::{variant}"))
}

fn relation_kind_expr(stage_type_prefix: &str, relation: &str) -> Result<String, EmitError> {
    let variant = JoltVerifierRelationKind::from_cpu_attr(relation)
        .map_err(plan_error)?
        .rust_variant();
    Ok(format!("{stage_type_prefix}RelationKind::{variant}"))
}

fn field_expr_kind_expr(stage_type_prefix: &str, formula: &str) -> Result<String, EmitError> {
    let variant = FieldExprKind::from_cpu_attr(formula)
        .map_err(plan_error)?
        .rust_variant_expr();
    Ok(format!("{stage_type_prefix}FieldExprKind::{variant}"))
}

fn sumcheck_point_order_expr(point_order: &str) -> Result<String, EmitError> {
    let variant = SumcheckPointOrder::from_cpu_attr(point_order)
        .map_err(plan_error)?
        .rust_variant();
    Ok(format!(
        "bolt_verifier_runtime::SumcheckPointOrder::{variant}"
    ))
}

pub(super) fn pcs_proof_mode_expr(
    stage_type_prefix: &str,
    mode: &str,
) -> Result<String, EmitError> {
    let variant = PcsProofMode::from_cpu_attr(mode)
        .map_err(plan_error)?
        .rust_variant();
    Ok(format!("{stage_type_prefix}PcsProofMode::{variant}"))
}

fn opening_equality_mode_expr(stage_type_prefix: &str, mode: &str) -> Result<String, EmitError> {
    let variant = OpeningEqualityMode::from_cpu_attr(mode)
        .map_err(plan_error)?
        .rust_variant();
    Ok(format!("{stage_type_prefix}OpeningEqualityMode::{variant}"))
}

fn plan_error(error: RustTargetPlanError) -> EmitError {
    EmitError::new(error.to_string())
}
