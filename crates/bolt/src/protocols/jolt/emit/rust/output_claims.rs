use std::collections::{BTreeMap, BTreeSet};

use crate::emit::rust::{push_format, EmitError};
use crate::ir::Role;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuredPolynomialPointPlan {
    pub source: String,
    pub segment: String,
    pub length: String,
    pub order: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuredPolynomialEvalPlan {
    pub symbol: String,
    pub polynomial: String,
    pub x_point: StructuredPolynomialPointPlan,
    pub y_point: StructuredPolynomialPointPlan,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckOutputClaimPlan {
    pub relation: String,
    pub polynomial_evals: Vec<StructuredPolynomialEvalPlan>,
    pub claim_value: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckOutputClaimAst {
    pub relation: String,
    pub polynomial_evals: Vec<String>,
    pub polynomial_eval_operands: Vec<String>,
    pub claim_value: String,
}

pub trait FieldExprDependencies {
    fn symbol(&self) -> &str;
    fn operands(&self) -> &[String];
}

pub fn resolve_output_claims(
    stage: &str,
    output_values: &[StructuredPolynomialEvalPlan],
    claim_asts: Vec<SumcheckOutputClaimAst>,
) -> Result<Vec<SumcheckOutputClaimPlan>, EmitError> {
    let output_values_by_symbol: BTreeMap<_, _> = output_values
        .iter()
        .map(|value| (value.symbol.as_str(), value))
        .collect();
    claim_asts
        .into_iter()
        .map(|claim| {
            verify_count(
                "sumcheck output claim polynomial_evals",
                &claim.relation,
                claim.polynomial_evals.len(),
                claim.polynomial_eval_operands.len(),
            )?;
            if claim.polynomial_evals != claim.polynomial_eval_operands {
                return Err(EmitError::new(format!(
                    "{stage} output claim for @{} polynomial_evals do not match operands",
                    claim.relation
                )));
            }
            let polynomial_evals = claim
                .polynomial_evals
                .iter()
                .map(|symbol| {
                    output_values_by_symbol
                        .get(symbol.as_str())
                        .copied()
                        .cloned()
                        .ok_or_else(|| {
                            EmitError::new(format!(
                                "{stage} output claim for @{} references missing output value @{symbol}",
                                claim.relation
                            ))
                        })
                })
                .collect::<Result<Vec<_>, EmitError>>()?;
            Ok(SumcheckOutputClaimPlan {
                relation: claim.relation,
                polynomial_evals,
                claim_value: claim.claim_value,
            })
        })
        .collect()
}

pub fn prune_output_only_field_exprs<'a, 'b, T>(
    field_exprs: &mut Vec<T>,
    sumcheck_claim_roots: impl Iterator<Item = &'a str>,
    output_claim_roots: impl Iterator<Item = &'b str>,
) where
    T: FieldExprDependencies,
{
    let field_exprs_by_symbol: BTreeMap<_, _> = field_exprs
        .iter()
        .map(|expr| (expr.symbol(), expr))
        .collect();
    let sumcheck_claim_closure =
        field_expr_dependency_closure(&field_exprs_by_symbol, sumcheck_claim_roots);
    let output_claim_closure =
        field_expr_dependency_closure(&field_exprs_by_symbol, output_claim_roots);
    field_exprs.retain(|expr| {
        !output_claim_closure.contains(expr.symbol())
            || sumcheck_claim_closure.contains(expr.symbol())
    });
}

pub fn verify_output_claims(
    stage: &str,
    output_values: &[StructuredPolynomialEvalPlan],
    output_claims: &[SumcheckOutputClaimPlan],
    relations: &BTreeSet<String>,
    field_values: &BTreeSet<String>,
    point_values: &BTreeSet<String>,
) -> Result<(), EmitError> {
    for polynomial_eval in output_values {
        if !point_values.contains(&polynomial_eval.x_point.source) {
            return Err(EmitError::new(format!(
                "{stage} structured polynomial eval @{} references missing x-point @{}",
                polynomial_eval.symbol, polynomial_eval.x_point.source
            )));
        }
        if !point_values.contains(&polynomial_eval.y_point.source) {
            return Err(EmitError::new(format!(
                "{stage} structured polynomial eval @{} references missing y-point @{}",
                polynomial_eval.symbol, polynomial_eval.y_point.source
            )));
        }
        if !matches!(
            polynomial_eval.polynomial.as_str(),
            "eq" | "eq_plus_one" | "lt"
        ) {
            return Err(EmitError::new(format!(
                "{stage} structured polynomial eval @{} has unsupported polynomial `{}`",
                polynomial_eval.symbol, polynomial_eval.polynomial
            )));
        }
        verify_structured_polynomial_point_plan(stage, polynomial_eval, &polynomial_eval.x_point)?;
        verify_structured_polynomial_point_plan(stage, polynomial_eval, &polynomial_eval.y_point)?;
    }
    for claim in output_claims {
        if !relations.contains(&claim.relation) {
            return Err(EmitError::new(format!(
                "{stage} output claim references missing relation @{}",
                claim.relation
            )));
        }
        if !field_values.contains(&claim.claim_value) {
            return Err(EmitError::new(format!(
                "{stage} output claim for @{} uses missing claim value @{}",
                claim.relation, claim.claim_value
            )));
        }
    }
    Ok(())
}

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
                    structured_polynomial_kind_expr(stage_type, &value.polynomial)?,
                    structured_polynomial_point_expr(stage_type, &value.x_point)?,
                    structured_polynomial_point_expr(stage_type, &value.y_point)?,
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
        claims.push(format!(
            "    {stage_type}SumcheckOutputClaimPlan {{ relation: {}, polynomial_evals: {values_name}, claim_value: {} }},",
            super::plan_tokens::role_relation_kind_expr(stage_type, role, &claim.relation)?,
            rust_str(&claim.claim_value)
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

fn field_expr_dependency_closure<'a, T>(
    field_exprs_by_symbol: &BTreeMap<&str, &T>,
    roots: impl Iterator<Item = &'a str>,
) -> BTreeSet<String>
where
    T: FieldExprDependencies,
{
    let mut visited = BTreeSet::new();
    let mut stack = roots.map(str::to_owned).collect::<Vec<_>>();
    while let Some(symbol) = stack.pop() {
        if !visited.insert(symbol.clone()) {
            continue;
        }
        let Some(expr) = field_exprs_by_symbol.get(symbol.as_str()) else {
            continue;
        };
        for operand in expr.operands() {
            if field_exprs_by_symbol.contains_key(operand.as_str()) {
                stack.push(operand.clone());
            }
        }
    }
    visited
}

fn verify_structured_polynomial_point_plan(
    stage: &str,
    polynomial_eval: &StructuredPolynomialEvalPlan,
    point: &StructuredPolynomialPointPlan,
) -> Result<(), EmitError> {
    if !matches!(point.segment.as_str(), "full" | "prefix" | "suffix") {
        return Err(EmitError::new(format!(
            "{stage} structured polynomial eval @{} has unsupported point segment `{}`",
            polynomial_eval.symbol, point.segment
        )));
    }
    if !matches!(point.length.as_str(), "full" | "x_point" | "y_point") {
        return Err(EmitError::new(format!(
            "{stage} structured polynomial eval @{} has unsupported point length `{}`",
            polynomial_eval.symbol, point.length
        )));
    }
    if !matches!(point.order.as_str(), "as_is" | "reverse") {
        return Err(EmitError::new(format!(
            "{stage} structured polynomial eval @{} has unsupported point order `{}`",
            polynomial_eval.symbol, point.order
        )));
    }
    Ok(())
}

fn structured_polynomial_kind_expr(
    stage_type: &str,
    polynomial: &str,
) -> Result<String, EmitError> {
    let variant = match polynomial {
        "eq" => "Eq",
        "eq_plus_one" => "EqPlusOne",
        "lt" => "Lt",
        _ => {
            return Err(EmitError::new(format!(
                "unsupported {stage_type} structured polynomial `{polynomial}`"
            )))
        }
    };
    Ok(format!("{stage_type}StructuredPolynomialKind::{variant}"))
}

fn structured_polynomial_point_expr(
    stage_type: &str,
    point: &StructuredPolynomialPointPlan,
) -> Result<String, EmitError> {
    Ok(format!(
        "{stage_type}StructuredPolynomialPointPlan {{ source: {}, segment: {}, length: {}, order: {} }}",
        rust_str(&point.source),
        structured_polynomial_point_segment_expr(stage_type, &point.segment)?,
        structured_polynomial_point_length_expr(stage_type, &point.length)?,
        structured_polynomial_point_order_expr(stage_type, &point.order)?,
    ))
}

fn structured_polynomial_point_segment_expr(
    stage_type: &str,
    segment: &str,
) -> Result<String, EmitError> {
    let variant = match segment {
        "full" => "Full",
        "prefix" => "Prefix",
        "suffix" => "Suffix",
        _ => {
            return Err(EmitError::new(format!(
                "unsupported {stage_type} output point segment `{segment}`"
            )))
        }
    };
    Ok(format!(
        "{stage_type}StructuredPolynomialPointSegment::{variant}"
    ))
}

fn structured_polynomial_point_length_expr(
    stage_type: &str,
    length: &str,
) -> Result<String, EmitError> {
    let variant = match length {
        "full" => "Full",
        "x_point" => "XPoint",
        "y_point" => "YPoint",
        _ => {
            return Err(EmitError::new(format!(
                "unsupported {stage_type} structured polynomial point length `{length}`"
            )))
        }
    };
    Ok(format!(
        "{stage_type}StructuredPolynomialPointLength::{variant}"
    ))
}

fn structured_polynomial_point_order_expr(
    stage_type: &str,
    order: &str,
) -> Result<String, EmitError> {
    let variant = match order {
        "as_is" => "AsIs",
        "reverse" => "Reverse",
        _ => {
            return Err(EmitError::new(format!(
                "unsupported {stage_type} output point order `{order}`"
            )))
        }
    };
    Ok(format!(
        "{stage_type}StructuredPolynomialPointOrder::{variant}"
    ))
}

fn verify_count(kind: &str, symbol: &str, expected: usize, actual: usize) -> Result<(), EmitError> {
    if expected == actual {
        Ok(())
    } else {
        Err(EmitError::new(format!(
            "{kind} @{symbol} count mismatch: expected {expected}, got {actual}"
        )))
    }
}

fn rust_str(value: &str) -> String {
    format!("{value:?}")
}
