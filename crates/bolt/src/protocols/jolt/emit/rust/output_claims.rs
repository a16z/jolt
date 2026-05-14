use std::collections::{BTreeMap, BTreeSet};

use melior::ir::operation::{OperationLike, OperationResult};
use melior::ir::{Attribute, OperationRef};

use crate::emit::rust::{push_format, EmitError};
use crate::ir::{string_attribute_value, Role};
use crate::schema::operation_name;

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
pub struct SumcheckOutputEvalFamilySharedTermPlan {
    pub gamma_power_offset: usize,
    pub factor: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckOutputEvalFamilyItemTermPlan {
    pub gamma_power_offset: usize,
    pub factors: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckOutputEvalFamilyPlan {
    pub symbol: String,
    pub gamma: String,
    pub evals: Vec<String>,
    pub power_stride: usize,
    pub value_term_offsets: Vec<usize>,
    pub shared_terms: Vec<SumcheckOutputEvalFamilySharedTermPlan>,
    pub item_terms: Vec<SumcheckOutputEvalFamilyItemTermPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckOutputProductFamilyTermPlan {
    pub gamma_power_offset: usize,
    pub evals: Vec<String>,
    pub factors: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckOutputProductFamilyPlan {
    pub symbol: String,
    pub gamma: String,
    pub terms: Vec<SumcheckOutputProductFamilyTermPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckOutputClaimPlan {
    pub relation: String,
    pub polynomial_evals: Vec<StructuredPolynomialEvalPlan>,
    pub eval_families: Vec<SumcheckOutputEvalFamilyPlan>,
    pub product_families: Vec<SumcheckOutputProductFamilyPlan>,
    pub claim_value: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckOutputClaimAst {
    pub relation: String,
    pub polynomial_evals: Vec<String>,
    pub polynomial_eval_operands: Vec<String>,
    pub claim_value: String,
}

pub fn parse_output_eval_family_plan(
    stage: &str,
    operation: OperationRef<'_, '_>,
) -> Result<SumcheckOutputEvalFamilyPlan, EmitError> {
    let symbol = string_attr(operation, "sym_name")?;
    let evals = symbol_array_attr(operation, "evals")?;
    let shared_factors = symbol_array_attr(operation, "shared_terms")?;
    let item_factors = symbol_array_attr(operation, "item_terms")?;
    let value_term_offsets = int_array_attr(operation, "value_term_offsets")?;
    let shared_term_offsets = int_array_attr(operation, "shared_term_offsets")?;
    let item_term_offsets = int_array_attr(operation, "item_term_offsets")?;
    verify_count(
        "output eval family shared terms",
        &symbol,
        shared_term_offsets.len(),
        shared_factors.len(),
    )?;
    verify_count(
        "output eval family item term factors",
        &symbol,
        item_term_offsets.len() * evals.len(),
        item_factors.len(),
    )?;
    let gamma = operand_symbol(operation, 0)?;
    let eval_operands = operand_symbols(operation, 1, 1 + evals.len())?;
    if evals != eval_operands {
        return Err(EmitError::new(format!(
            "{stage} output eval family @{symbol} evals do not match operands"
        )));
    }
    let shared_start = 1 + evals.len();
    let shared_end = shared_start + shared_factors.len();
    let shared_operands = operand_symbols(operation, shared_start, shared_end)?;
    if shared_factors != shared_operands {
        return Err(EmitError::new(format!(
            "{stage} output eval family @{symbol} shared_terms do not match operands"
        )));
    }
    let item_operands = operand_symbols(operation, shared_end, operation.operand_count())?;
    if item_factors != item_operands {
        return Err(EmitError::new(format!(
            "{stage} output eval family @{symbol} item_terms do not match operands"
        )));
    }
    let shared_terms = shared_term_offsets
        .into_iter()
        .zip(shared_factors)
        .map(
            |(gamma_power_offset, factor)| SumcheckOutputEvalFamilySharedTermPlan {
                gamma_power_offset,
                factor,
            },
        )
        .collect();
    let mut item_terms = Vec::with_capacity(item_term_offsets.len());
    for (term_index, gamma_power_offset) in item_term_offsets.into_iter().enumerate() {
        let start = term_index * evals.len();
        let end = start + evals.len();
        item_terms.push(SumcheckOutputEvalFamilyItemTermPlan {
            gamma_power_offset,
            factors: item_factors[start..end].to_vec(),
        });
    }
    Ok(SumcheckOutputEvalFamilyPlan {
        symbol,
        gamma,
        evals,
        power_stride: int_attr(operation, "power_stride")?,
        value_term_offsets,
        shared_terms,
        item_terms,
    })
}

pub fn parse_output_product_family_plan(
    stage: &str,
    operation: OperationRef<'_, '_>,
) -> Result<SumcheckOutputProductFamilyPlan, EmitError> {
    let symbol = string_attr(operation, "sym_name")?;
    let evals = symbol_array_attr(operation, "evals")?;
    let factors = symbol_array_attr(operation, "factors")?;
    let term_gamma_power_offsets = int_array_attr(operation, "term_gamma_power_offsets")?;
    let term_eval_counts = int_array_attr(operation, "term_eval_counts")?;
    let term_factor_counts = int_array_attr(operation, "term_factor_counts")?;
    verify_count(
        "output product family term eval counts",
        &symbol,
        term_gamma_power_offsets.len(),
        term_eval_counts.len(),
    )?;
    verify_count(
        "output product family term factor counts",
        &symbol,
        term_gamma_power_offsets.len(),
        term_factor_counts.len(),
    )?;
    verify_count(
        "output product family evals",
        &symbol,
        term_eval_counts.iter().sum(),
        evals.len(),
    )?;
    verify_count(
        "output product family factors",
        &symbol,
        term_factor_counts.iter().sum(),
        factors.len(),
    )?;
    let gamma = operand_symbol(operation, 0)?;
    let eval_end = 1 + evals.len();
    let eval_operands = operand_symbols(operation, 1, eval_end)?;
    if evals != eval_operands {
        return Err(EmitError::new(format!(
            "{stage} output product family @{symbol} evals do not match operands"
        )));
    }
    let factor_operands = operand_symbols(operation, eval_end, operation.operand_count())?;
    if factors != factor_operands {
        return Err(EmitError::new(format!(
            "{stage} output product family @{symbol} factors do not match operands"
        )));
    }
    let mut eval_offset = 0;
    let mut factor_offset = 0;
    let mut terms = Vec::with_capacity(term_gamma_power_offsets.len());
    for ((gamma_power_offset, eval_count), factor_count) in term_gamma_power_offsets
        .into_iter()
        .zip(term_eval_counts)
        .zip(term_factor_counts)
    {
        if eval_count == 0 && factor_count == 0 {
            return Err(EmitError::new(format!(
                "{stage} output product family @{symbol} has an empty term"
            )));
        }
        let eval_end = eval_offset + eval_count;
        let factor_end = factor_offset + factor_count;
        terms.push(SumcheckOutputProductFamilyTermPlan {
            gamma_power_offset,
            evals: evals[eval_offset..eval_end].to_vec(),
            factors: factors[factor_offset..factor_end].to_vec(),
        });
        eval_offset = eval_end;
        factor_offset = factor_end;
    }
    Ok(SumcheckOutputProductFamilyPlan {
        symbol,
        gamma,
        terms,
    })
}

pub trait FieldExprDependencies {
    fn symbol(&self) -> &str;
    fn operands(&self) -> &[String];
}

pub fn resolve_output_claims<T>(
    stage: &str,
    output_values: &[StructuredPolynomialEvalPlan],
    output_families: &[SumcheckOutputEvalFamilyPlan],
    output_product_families: &[SumcheckOutputProductFamilyPlan],
    field_exprs: &[T],
    claim_asts: Vec<SumcheckOutputClaimAst>,
) -> Result<Vec<SumcheckOutputClaimPlan>, EmitError>
where
    T: FieldExprDependencies,
{
    let output_values_by_symbol: BTreeMap<_, _> = output_values
        .iter()
        .map(|value| (value.symbol.as_str(), value))
        .collect();
    let output_families_by_symbol: BTreeMap<_, _> = output_families
        .iter()
        .map(|family| (family.symbol.as_str(), family))
        .collect();
    let output_product_families_by_symbol: BTreeMap<_, _> = output_product_families
        .iter()
        .map(|family| (family.symbol.as_str(), family))
        .collect();
    let field_exprs_by_symbol: BTreeMap<_, _> = field_exprs
        .iter()
        .map(|expr| (expr.symbol(), expr))
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
            let family_symbols = output_family_dependency_closure(
                &output_families_by_symbol,
                &output_product_families_by_symbol,
                &field_exprs_by_symbol,
                std::iter::once(claim.claim_value.as_str()),
            );
            let eval_families = output_families
                .iter()
                .filter(|family| family_symbols.eval_families.contains(&family.symbol))
                .cloned()
                .collect();
            let product_families = output_product_families
                .iter()
                .filter(|family| family_symbols.product_families.contains(&family.symbol))
                .cloned()
                .collect();
            Ok(SumcheckOutputClaimPlan {
                relation: claim.relation,
                polynomial_evals,
                eval_families,
                product_families,
                claim_value: claim.claim_value,
            })
        })
        .collect()
}

struct OutputFamilyDependencyClosure {
    eval_families: BTreeSet<String>,
    product_families: BTreeSet<String>,
}

fn output_family_dependency_closure<'a, T>(
    output_families_by_symbol: &BTreeMap<&str, &SumcheckOutputEvalFamilyPlan>,
    output_product_families_by_symbol: &BTreeMap<&str, &SumcheckOutputProductFamilyPlan>,
    field_exprs_by_symbol: &BTreeMap<&str, &T>,
    roots: impl Iterator<Item = &'a str>,
) -> OutputFamilyDependencyClosure
where
    T: FieldExprDependencies,
{
    let mut visited = BTreeSet::new();
    let mut eval_families = BTreeSet::new();
    let mut product_families = BTreeSet::new();
    let mut stack = roots.map(str::to_owned).collect::<Vec<_>>();
    while let Some(symbol) = stack.pop() {
        if !visited.insert(symbol.clone()) {
            continue;
        }
        if let Some(family) = output_families_by_symbol.get(symbol.as_str()) {
            let _inserted = eval_families.insert(family.symbol.clone());
            stack.push(family.gamma.clone());
            stack.extend(family.evals.iter().cloned());
            stack.extend(family.shared_terms.iter().map(|term| term.factor.clone()));
            stack.extend(
                family
                    .item_terms
                    .iter()
                    .flat_map(|term| term.factors.iter().cloned()),
            );
        }
        if let Some(family) = output_product_families_by_symbol.get(symbol.as_str()) {
            let _inserted = product_families.insert(family.symbol.clone());
            stack.push(family.gamma.clone());
            for term in &family.terms {
                stack.extend(term.evals.iter().cloned());
                stack.extend(term.factors.iter().cloned());
            }
        }
        if let Some(expr) = field_exprs_by_symbol.get(symbol.as_str()) {
            stack.extend(expr.operands().iter().cloned());
        }
    }
    OutputFamilyDependencyClosure {
        eval_families,
        product_families,
    }
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

pub struct OutputClaimVerification<'a> {
    pub output_values: &'a [StructuredPolynomialEvalPlan],
    pub output_families: &'a [SumcheckOutputEvalFamilyPlan],
    pub output_product_families: &'a [SumcheckOutputProductFamilyPlan],
    pub output_claims: &'a [SumcheckOutputClaimPlan],
    pub relations: &'a BTreeSet<String>,
    pub field_values: &'a BTreeSet<String>,
    pub point_values: &'a BTreeSet<String>,
}

pub fn verify_output_claims(
    stage: &str,
    verification: OutputClaimVerification<'_>,
) -> Result<(), EmitError> {
    let OutputClaimVerification {
        output_values,
        output_families,
        output_product_families,
        output_claims,
        relations,
        field_values,
        point_values,
    } = verification;
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
    for family in output_families {
        if !field_values.contains(&family.gamma) {
            return Err(EmitError::new(format!(
                "{stage} output eval family @{} references missing gamma @{}",
                family.symbol, family.gamma
            )));
        }
        for eval in &family.evals {
            if !field_values.contains(eval) {
                return Err(EmitError::new(format!(
                    "{stage} output eval family @{} references missing eval @{}",
                    family.symbol, eval
                )));
            }
        }
        for term in &family.shared_terms {
            if !field_values.contains(&term.factor) {
                return Err(EmitError::new(format!(
                    "{stage} output eval family @{} references missing shared factor @{}",
                    family.symbol, term.factor
                )));
            }
        }
        for term in &family.item_terms {
            verify_count(
                "output eval family item factors",
                &family.symbol,
                family.evals.len(),
                term.factors.len(),
            )?;
            for factor in &term.factors {
                if !field_values.contains(factor) {
                    return Err(EmitError::new(format!(
                        "{stage} output eval family @{} references missing item factor @{}",
                        family.symbol, factor
                    )));
                }
            }
        }
    }
    for family in output_product_families {
        if !field_values.contains(&family.gamma) {
            return Err(EmitError::new(format!(
                "{stage} output product family @{} references missing gamma @{}",
                family.symbol, family.gamma
            )));
        }
        for term in &family.terms {
            if term.evals.is_empty() && term.factors.is_empty() {
                return Err(EmitError::new(format!(
                    "{stage} output product family @{} has an empty term",
                    family.symbol
                )));
            }
            for eval in &term.evals {
                if !field_values.contains(eval) {
                    return Err(EmitError::new(format!(
                        "{stage} output product family @{} references missing eval @{}",
                        family.symbol, eval
                    )));
                }
            }
            for factor in &term.factors {
                if !field_values.contains(factor) {
                    return Err(EmitError::new(format!(
                        "{stage} output product family @{} references missing factor @{}",
                        family.symbol, factor
                    )));
                }
            }
        }
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
        let eval_families = emit_eval_family_constants(&mut source, stage_type, index, claim);
        let product_families = emit_product_family_constants(&mut source, stage_type, index, claim);
        claims.push(format!(
            "    {stage_type}SumcheckOutputClaimPlan {{ relation: {}, polynomial_evals: {values_name}, eval_families: {eval_families}, product_families: {product_families}, claim_value: {} }},",
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
        let value_offsets = usize_array(&family.value_term_offsets);
        push_format(
            source,
            format_args!("pub const {value_offsets_name}: &[usize] = &[{value_offsets}];\n"),
        );
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
            let factors = rust_str_array(&term.factors);
            push_format(
                source,
                format_args!("pub const {factors_name}: &[&str] = &[{factors}];\n"),
            );
            item_rows.push(format!(
                "    bolt_verifier_runtime::SumcheckOutputEvalFamilyItemTermPlan {{ gamma_power_offset: {}, factors: {factors_name} }},",
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
            "    bolt_verifier_runtime::SumcheckOutputEvalFamilyPlan {{ symbol: {}, gamma: {}, evals: {evals_name}, power_stride: {}, value_term_offsets: {value_offsets_name}, shared_terms: {shared_terms_name}, item_terms: {item_terms_name} }},",
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
            let evals = rust_str_array(&term.evals);
            push_format(
                source,
                format_args!("pub const {evals_name}: &[&str] = &[{evals}];\n"),
            );
            let factors_name = format!("{prefix}_TERM_{term_index}_FACTORS");
            let factors = rust_str_array(&term.factors);
            push_format(
                source,
                format_args!("pub const {factors_name}: &[&str] = &[{factors}];\n"),
            );
            term_rows.push(format!(
                "    bolt_verifier_runtime::SumcheckOutputProductFamilyTermPlan {{ gamma_power_offset: {}, evals: {evals_name}, factors: {factors_name} }},",
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
            rust_str(&family.gamma),
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

fn string_attr(operation: OperationRef<'_, '_>, attr: &str) -> Result<String, EmitError> {
    operation
        .attribute(attr)
        .ok()
        .and_then(string_attribute_value)
        .ok_or_else(|| attr_error(operation, attr, "string"))
}

fn symbol_array_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<Vec<String>, EmitError> {
    let attribute = operation
        .attribute(attr)
        .map(|attribute| attribute.to_string())
        .ok()
        .ok_or_else(|| attr_error(operation, attr, "symbol array"))?;
    parse_symbol_array(&attribute).ok_or_else(|| attr_error(operation, attr, "symbol array"))
}

fn parse_symbol_array(attribute: &str) -> Option<Vec<String>> {
    let inner = attribute.strip_prefix('[')?.strip_suffix(']')?.trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    inner
        .split(',')
        .map(|item| item.trim().strip_prefix('@').map(ToOwned::to_owned))
        .collect()
}

fn int_attr(operation: OperationRef<'_, '_>, attr: &str) -> Result<usize, EmitError> {
    operation
        .attribute(attr)
        .map(parse_integer_attr)
        .ok()
        .flatten()
        .ok_or_else(|| attr_error(operation, attr, "integer"))
}

fn parse_integer_attr(attribute: Attribute<'_>) -> Option<usize> {
    attribute
        .to_string()
        .split_whitespace()
        .next()
        .and_then(|value| value.parse().ok())
}

fn int_array_attr(operation: OperationRef<'_, '_>, attr: &str) -> Result<Vec<usize>, EmitError> {
    let attribute = operation
        .attribute(attr)
        .map(|attribute| attribute.to_string())
        .ok()
        .ok_or_else(|| attr_error(operation, attr, "integer array"))?;
    parse_int_array(&attribute).ok_or_else(|| attr_error(operation, attr, "integer array"))
}

fn parse_int_array(attribute: &str) -> Option<Vec<usize>> {
    let inner = attribute.strip_prefix('[')?.strip_suffix(']')?.trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    inner
        .split(',')
        .map(|item| item.trim().parse().ok())
        .collect()
}

fn operand_symbols(
    operation: OperationRef<'_, '_>,
    start_index: usize,
    end_index: usize,
) -> Result<Vec<String>, EmitError> {
    (start_index..end_index)
        .map(|index| operand_symbol(operation, index))
        .collect()
}

fn operand_symbol(operation: OperationRef<'_, '_>, index: usize) -> Result<String, EmitError> {
    let operand = operation.operand(index).map_err(|_| {
        EmitError::new(format!(
            "{} requires operand {index}",
            operation_name(operation)
        ))
    })?;
    let owner = OperationResult::try_from(operand).map_err(|_| {
        EmitError::new(format!(
            "{} operand {index} must be an op result",
            operation_name(operation)
        ))
    })?;
    string_attr(owner.owner(), "sym_name")
}

fn attr_error(operation: OperationRef<'_, '_>, attr: &str, expected: &str) -> EmitError {
    EmitError::new(format!(
        "{} attr `{attr}` is not a {expected}",
        operation_name(operation)
    ))
}

#[cfg(test)]
mod tests {
    use crate::emit::rust::EmitError;

    use super::{
        resolve_output_claims, FieldExprDependencies, SumcheckOutputClaimAst,
        SumcheckOutputEvalFamilyItemTermPlan, SumcheckOutputEvalFamilyPlan,
        SumcheckOutputEvalFamilySharedTermPlan, SumcheckOutputProductFamilyPlan,
        SumcheckOutputProductFamilyTermPlan,
    };

    struct TestFieldExpr {
        symbol: String,
        operands: Vec<String>,
    }

    impl FieldExprDependencies for TestFieldExpr {
        fn symbol(&self) -> &str {
            &self.symbol
        }

        fn operands(&self) -> &[String] {
            &self.operands
        }
    }

    #[test]
    fn resolves_output_families_reachable_through_field_expressions() -> Result<(), EmitError> {
        let inner_family = SumcheckOutputEvalFamilyPlan {
            symbol: "inner.family".to_owned(),
            gamma: "inner.gamma".to_owned(),
            evals: vec!["inner.eval".to_owned()],
            power_stride: 1,
            value_term_offsets: vec![0],
            shared_terms: Vec::new(),
            item_terms: Vec::new(),
        };
        let outer_family = SumcheckOutputEvalFamilyPlan {
            symbol: "outer.family".to_owned(),
            gamma: "outer.gamma".to_owned(),
            evals: vec!["outer.eval".to_owned()],
            power_stride: 1,
            value_term_offsets: Vec::new(),
            shared_terms: vec![SumcheckOutputEvalFamilySharedTermPlan {
                gamma_power_offset: 0,
                factor: "outer.shared.factor".to_owned(),
            }],
            item_terms: vec![SumcheckOutputEvalFamilyItemTermPlan {
                gamma_power_offset: 1,
                factors: vec!["factor.expr".to_owned()],
            }],
        };
        let field_exprs = vec![
            TestFieldExpr {
                symbol: "claim.expr".to_owned(),
                operands: vec!["outer.family".to_owned()],
            },
            TestFieldExpr {
                symbol: "factor.expr".to_owned(),
                operands: vec!["inner.family".to_owned()],
            },
        ];
        let claim_asts = vec![SumcheckOutputClaimAst {
            relation: "relation".to_owned(),
            polynomial_evals: Vec::new(),
            polynomial_eval_operands: Vec::new(),
            claim_value: "claim.expr".to_owned(),
        }];

        let claims = resolve_output_claims(
            "test",
            &[],
            &[inner_family, outer_family],
            &[],
            &field_exprs,
            claim_asts,
        )?;
        let claim = claims
            .first()
            .ok_or_else(|| EmitError::new("missing resolved output claim"))?;
        let family_symbols = claim
            .eval_families
            .iter()
            .map(|family| family.symbol.as_str())
            .collect::<Vec<_>>();
        assert_eq!(family_symbols, vec!["inner.family", "outer.family"]);
        assert!(claim.product_families.is_empty());
        Ok(())
    }

    #[test]
    fn resolves_product_families_reachable_through_field_expressions() -> Result<(), EmitError> {
        let product_family = SumcheckOutputProductFamilyPlan {
            symbol: "product.family".to_owned(),
            gamma: "product.gamma".to_owned(),
            terms: vec![SumcheckOutputProductFamilyTermPlan {
                gamma_power_offset: 2,
                evals: vec!["product.eval".to_owned()],
                factors: vec!["product.factor.expr".to_owned()],
            }],
        };
        let field_exprs = vec![
            TestFieldExpr {
                symbol: "claim.expr".to_owned(),
                operands: vec!["product.family".to_owned()],
            },
            TestFieldExpr {
                symbol: "product.factor.expr".to_owned(),
                operands: vec!["unrelated.scalar".to_owned()],
            },
        ];
        let claim_asts = vec![SumcheckOutputClaimAst {
            relation: "relation".to_owned(),
            polynomial_evals: Vec::new(),
            polynomial_eval_operands: Vec::new(),
            claim_value: "claim.expr".to_owned(),
        }];

        let claims = resolve_output_claims(
            "test",
            &[],
            &[],
            &[product_family],
            &field_exprs,
            claim_asts,
        )?;
        let claim = claims
            .first()
            .ok_or_else(|| EmitError::new("missing resolved output claim"))?;
        assert!(claim.eval_families.is_empty());
        let family_symbols = claim
            .product_families
            .iter()
            .map(|family| family.symbol.as_str())
            .collect::<Vec<_>>();
        assert_eq!(family_symbols, vec!["product.family"]);
        Ok(())
    }
}
