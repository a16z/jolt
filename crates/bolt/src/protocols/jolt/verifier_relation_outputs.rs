use std::collections::{BTreeMap, BTreeSet};

use melior::ir::operation::{OperationLike, OperationResult};
use melior::ir::OperationRef;

use crate::emit::rust::EmitError;
use crate::protocols::jolt::cpu_attrs::{
    int_array_attr, int_attr, operation_name, optional_symbol_array_attr, string_array_attr,
    string_attr, symbol_array_attr,
};
use crate::protocols::jolt::rust_target_plan::{
    power_strided_weighted_sum_formula, structured_polynomial_scalar_formula,
};
use crate::protocols::jolt::verifier_values::{
    VerifierPointSourceSet, VerifierScalarValueKind, VerifierScalarValuePlan,
    VerifierScalarValueRef, VerifierScalarValueSet,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructuredPolynomialKind {
    Eq,
    EqPlusOne,
    Lt,
}

impl StructuredPolynomialKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::EqPlusOne => "eq_plus_one",
            Self::Lt => "lt",
        }
    }

    pub fn from_cpu_attr(value: &str) -> Result<Self, EmitError> {
        match value {
            "eq" => Ok(Self::Eq),
            "eq_plus_one" => Ok(Self::EqPlusOne),
            "lt" => Ok(Self::Lt),
            _ => Err(EmitError::new(format!(
                "unsupported structured polynomial `{value}`"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructuredPolynomialPointSegment {
    Full,
    Prefix,
    Suffix,
}

impl StructuredPolynomialPointSegment {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Prefix => "prefix",
            Self::Suffix => "suffix",
        }
    }

    pub fn from_cpu_attr(value: &str) -> Result<Self, EmitError> {
        match value {
            "full" => Ok(Self::Full),
            "prefix" => Ok(Self::Prefix),
            "suffix" => Ok(Self::Suffix),
            _ => Err(EmitError::new(format!(
                "unsupported structured polynomial point segment `{value}`"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructuredPolynomialPointLength {
    Full,
    XPoint,
    YPoint,
}

impl StructuredPolynomialPointLength {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::XPoint => "x_point",
            Self::YPoint => "y_point",
        }
    }

    pub fn from_cpu_attr(value: &str) -> Result<Self, EmitError> {
        match value {
            "full" => Ok(Self::Full),
            "x_point" => Ok(Self::XPoint),
            "y_point" => Ok(Self::YPoint),
            _ => Err(EmitError::new(format!(
                "unsupported structured polynomial point length `{value}`"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructuredPolynomialPointOrder {
    AsIs,
    Reverse,
}

impl StructuredPolynomialPointOrder {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AsIs => "as_is",
            Self::Reverse => "reverse",
        }
    }

    pub fn from_cpu_attr(value: &str) -> Result<Self, EmitError> {
        match value {
            "as_is" => Ok(Self::AsIs),
            "reverse" => Ok(Self::Reverse),
            _ => Err(EmitError::new(format!(
                "unsupported structured polynomial point order `{value}`"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelationOutputFunctionKind {
    BooleanZero,
}

impl RelationOutputFunctionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BooleanZero => "boolean_zero",
        }
    }

    pub fn from_cpu_attr(value: &str) -> Result<Self, EmitError> {
        match value {
            "boolean_zero" => Ok(Self::BooleanZero),
            _ => Err(EmitError::new(format!(
                "unsupported relation output function `{value}`"
            ))),
        }
    }
}

impl PartialEq<&str> for RelationOutputFunctionKind {
    fn eq(&self, other: &&str) -> bool {
        matches!((self, *other), (Self::BooleanZero, "boolean_zero"))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuredPolynomialPointPlan {
    pub source: String,
    pub segment: StructuredPolynomialPointSegment,
    pub length: StructuredPolynomialPointLength,
    pub order: StructuredPolynomialPointOrder,
}

impl StructuredPolynomialPointPlan {
    pub fn from_cpu(
        source: String,
        segment: String,
        length: String,
        order: String,
    ) -> Result<Self, EmitError> {
        Ok(Self {
            source,
            segment: StructuredPolynomialPointSegment::from_cpu_attr(&segment)?,
            length: StructuredPolynomialPointLength::from_cpu_attr(&length)?,
            order: StructuredPolynomialPointOrder::from_cpu_attr(&order)?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuredPolynomialEvalPlan {
    pub symbol: String,
    pub polynomial: StructuredPolynomialKind,
    pub x_point: StructuredPolynomialPointPlan,
    pub y_point: StructuredPolynomialPointPlan,
}

impl StructuredPolynomialEvalPlan {
    pub fn from_cpu(
        symbol: String,
        polynomial: String,
        x_point: StructuredPolynomialPointPlan,
        y_point: StructuredPolynomialPointPlan,
    ) -> Result<Self, EmitError> {
        Ok(Self {
            symbol,
            polynomial: StructuredPolynomialKind::from_cpu_attr(&polynomial)?,
            x_point,
            y_point,
        })
    }
}

pub fn parse_structured_polynomial_eval_plan(
    operation: OperationRef<'_, '_>,
) -> Result<StructuredPolynomialEvalPlan, EmitError> {
    let x_point = StructuredPolynomialPointPlan::from_cpu(
        operand_symbol(operation, 0)?,
        string_attr(operation, "x_point_segment")?,
        string_attr(operation, "x_point_length")?,
        string_attr(operation, "x_point_order")?,
    )?;
    let y_point = StructuredPolynomialPointPlan::from_cpu(
        operand_symbol(operation, 1)?,
        string_attr(operation, "y_point_segment")?,
        string_attr(operation, "y_point_length")?,
        string_attr(operation, "y_point_order")?,
    )?;
    StructuredPolynomialEvalPlan::from_cpu(
        string_attr(operation, "sym_name")?,
        string_attr(operation, "polynomial")?,
        x_point,
        y_point,
    )
}

pub fn structured_polynomial_scalar_expr_plan(
    value: &StructuredPolynomialEvalPlan,
) -> RelationOutputFieldExprPlan {
    RelationOutputFieldExprPlan {
        symbol: value.symbol.clone(),
        formula: structured_polynomial_scalar_formula(
            value.polynomial.as_str(),
            value.x_point.segment.as_str(),
            value.x_point.length.as_str(),
            value.x_point.order.as_str(),
            value.y_point.segment.as_str(),
            value.y_point.length.as_str(),
            value.y_point.order.as_str(),
        ),
        operands: vec![value.x_point.source.clone(), value.y_point.source.clone()],
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelationOutputEvalFamilySharedTermPlan {
    pub gamma_power_offset: usize,
    pub factor: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelationOutputEvalFamilyItemTermPlan {
    pub gamma_power_offset: usize,
    pub factors: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelationOutputEvalFamilyPlan {
    pub symbol: String,
    pub gamma: String,
    pub evals: Vec<String>,
    pub power_stride: usize,
    pub value_term_offsets: Vec<usize>,
    pub shared_terms: Vec<RelationOutputEvalFamilySharedTermPlan>,
    pub item_terms: Vec<RelationOutputEvalFamilyItemTermPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelationOutputProductFamilyTermPlan {
    pub gamma_power_offset: usize,
    pub evals: Vec<String>,
    pub eval_families: Vec<String>,
    pub factors: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelationOutputProductFamilyPlan {
    pub symbol: String,
    pub gamma: Option<String>,
    pub terms: Vec<RelationOutputProductFamilyTermPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelationOutputFunctionFamilyTermPlan {
    pub gamma_power_offset: usize,
    pub function: RelationOutputFunctionKind,
    pub eval: String,
    pub factors: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelationOutputFunctionFamilyPlan {
    pub symbol: String,
    pub gamma: Option<String>,
    pub terms: Vec<RelationOutputFunctionFamilyTermPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelationOutputPlan {
    pub relation: String,
    local_scalars: Vec<VerifierScalarValuePlan>,
    expected_output: VerifierScalarValueRef,
}

impl RelationOutputPlan {
    pub fn new(relation: impl Into<String>, expected_output: impl Into<String>) -> Self {
        Self {
            relation: relation.into(),
            local_scalars: Vec::new(),
            expected_output: VerifierScalarValueRef::new(expected_output),
        }
    }

    pub(crate) fn with_local_scalars(
        relation: impl Into<String>,
        local_scalars: impl IntoIterator<Item = String>,
        expected_output: impl Into<String>,
    ) -> Self {
        Self {
            relation: relation.into(),
            local_scalars: local_scalars
                .into_iter()
                .map(|symbol| {
                    VerifierScalarValuePlan::new(
                        symbol,
                        VerifierScalarValueKind::RelationOutputLocal,
                    )
                })
                .collect(),
            expected_output: VerifierScalarValueRef::new(expected_output),
        }
    }

    pub fn local_scalar_symbols(&self) -> impl Iterator<Item = &String> {
        self.local_scalars.iter().map(|value| &value.symbol)
    }

    pub fn has_local_scalars(&self) -> bool {
        !self.local_scalars.is_empty()
    }

    pub fn expected_output_symbol(&self) -> &str {
        self.expected_output.symbol()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelationOutputAst {
    pub relation: String,
    pub polynomial_evals: Vec<String>,
    pub polynomial_eval_operands: Vec<String>,
    pub expected_output: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelationOutputFieldExprPlan {
    pub symbol: String,
    pub formula: String,
    pub operands: Vec<String>,
}

impl FieldExprDependencies for RelationOutputFieldExprPlan {
    fn symbol(&self) -> &str {
        &self.symbol
    }

    fn operands(&self) -> &[String] {
        &self.operands
    }
}

pub fn parse_output_eval_family_plan(
    stage: &str,
    operation: OperationRef<'_, '_>,
) -> Result<RelationOutputEvalFamilyPlan, EmitError> {
    let symbol = string_attr(operation, "sym_name")?;
    let evals = symbol_array_attr(operation, "evals")?;
    let shared_factors = symbol_array_attr(operation, "shared_terms")?;
    let item_factors = symbol_array_attr(operation, "item_terms")?;
    let value_term_offsets = int_array_attr(operation, "value_term_offsets")?;
    let shared_term_offsets = int_array_attr(operation, "shared_term_offsets")?;
    let item_term_offsets = int_array_attr(operation, "item_term_offsets")?;
    verify_count(
        "relation output eval family shared terms",
        &symbol,
        shared_term_offsets.len(),
        shared_factors.len(),
    )?;
    verify_count(
        "relation output eval family item term factors",
        &symbol,
        item_term_offsets.len() * evals.len(),
        item_factors.len(),
    )?;
    let gamma = operand_symbol(operation, 0)?;
    let eval_operands = operand_symbols(operation, 1, 1 + evals.len())?;
    if evals != eval_operands {
        return Err(EmitError::new(format!(
            "{stage} relation output eval family @{symbol} evals do not match operands"
        )));
    }
    let shared_start = 1 + evals.len();
    let shared_end = shared_start + shared_factors.len();
    let shared_operands = operand_symbols(operation, shared_start, shared_end)?;
    if shared_factors != shared_operands {
        return Err(EmitError::new(format!(
            "{stage} relation output eval family @{symbol} shared_terms do not match operands"
        )));
    }
    let item_operands = operand_symbols(operation, shared_end, operation.operand_count())?;
    if item_factors != item_operands {
        return Err(EmitError::new(format!(
            "{stage} relation output eval family @{symbol} item_terms do not match operands"
        )));
    }
    let shared_terms = shared_term_offsets
        .into_iter()
        .zip(shared_factors)
        .map(
            |(gamma_power_offset, factor)| RelationOutputEvalFamilySharedTermPlan {
                gamma_power_offset,
                factor,
            },
        )
        .collect();
    let mut item_terms = Vec::with_capacity(item_term_offsets.len());
    for (term_index, gamma_power_offset) in item_term_offsets.into_iter().enumerate() {
        let start = term_index * evals.len();
        let end = start + evals.len();
        item_terms.push(RelationOutputEvalFamilyItemTermPlan {
            gamma_power_offset,
            factors: item_factors[start..end].to_vec(),
        });
    }
    Ok(RelationOutputEvalFamilyPlan {
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
) -> Result<RelationOutputProductFamilyPlan, EmitError> {
    let symbol = string_attr(operation, "sym_name")?;
    let gamma = optional_symbol_array_attr(operation, "gamma")?;
    let evals = symbol_array_attr(operation, "evals")?;
    let factors = symbol_array_attr(operation, "factors")?;
    let term_gamma_power_offsets = int_array_attr(operation, "term_gamma_power_offsets")?;
    let term_eval_counts = int_array_attr(operation, "term_eval_counts")?;
    let term_factor_counts = int_array_attr(operation, "term_factor_counts")?;
    verify_count(
        "relation output product family term eval counts",
        &symbol,
        term_gamma_power_offsets.len(),
        term_eval_counts.len(),
    )?;
    verify_count(
        "relation output product family term factor counts",
        &symbol,
        term_gamma_power_offsets.len(),
        term_factor_counts.len(),
    )?;
    verify_count(
        "relation output product family evals",
        &symbol,
        term_eval_counts.iter().sum(),
        evals.len(),
    )?;
    verify_count(
        "relation output product family factors",
        &symbol,
        term_factor_counts.iter().sum(),
        factors.len(),
    )?;
    let eval_start = gamma.len();
    if let Some(gamma_symbol) = gamma.first() {
        let gamma_operand = operand_symbol(operation, 0)?;
        if gamma_operand != *gamma_symbol {
            return Err(EmitError::new(format!(
                "{stage} relation output product family @{symbol} gamma does not match operand"
            )));
        }
    }
    let eval_end = eval_start + evals.len();
    let eval_operands = operand_symbols(operation, eval_start, eval_end)?;
    if evals != eval_operands {
        return Err(EmitError::new(format!(
            "{stage} relation output product family @{symbol} evals do not match operands"
        )));
    }
    let factor_operands = operand_symbols(operation, eval_end, operation.operand_count())?;
    if factors != factor_operands {
        return Err(EmitError::new(format!(
            "{stage} relation output product family @{symbol} factors do not match operands"
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
                "{stage} relation output product family @{symbol} has an empty term"
            )));
        }
        let eval_end = eval_offset + eval_count;
        let factor_end = factor_offset + factor_count;
        terms.push(RelationOutputProductFamilyTermPlan {
            gamma_power_offset,
            evals: evals[eval_offset..eval_end].to_vec(),
            eval_families: Vec::new(),
            factors: factors[factor_offset..factor_end].to_vec(),
        });
        eval_offset = eval_end;
        factor_offset = factor_end;
    }
    Ok(RelationOutputProductFamilyPlan {
        symbol,
        gamma: gamma.into_iter().next(),
        terms,
    })
}

pub fn parse_output_function_family_plan(
    stage: &str,
    operation: OperationRef<'_, '_>,
) -> Result<RelationOutputFunctionFamilyPlan, EmitError> {
    let symbol = string_attr(operation, "sym_name")?;
    let gamma = optional_symbol_array_attr(operation, "gamma")?;
    let evals = symbol_array_attr(operation, "evals")?;
    let factors = symbol_array_attr(operation, "factors")?;
    let term_gamma_power_offsets = int_array_attr(operation, "term_gamma_power_offsets")?;
    let term_functions = string_array_attr(operation, "term_functions")?;
    let term_factor_counts = int_array_attr(operation, "term_factor_counts")?;
    verify_count(
        "relation output function family term functions",
        &symbol,
        term_gamma_power_offsets.len(),
        term_functions.len(),
    )?;
    verify_count(
        "relation output function family term factor counts",
        &symbol,
        term_gamma_power_offsets.len(),
        term_factor_counts.len(),
    )?;
    verify_count(
        "relation output function family evals",
        &symbol,
        term_gamma_power_offsets.len(),
        evals.len(),
    )?;
    verify_count(
        "relation output function family factors",
        &symbol,
        term_factor_counts.iter().sum(),
        factors.len(),
    )?;
    let eval_start = gamma.len();
    if let Some(gamma_symbol) = gamma.first() {
        let gamma_operand = operand_symbol(operation, 0)?;
        if gamma_operand != *gamma_symbol {
            return Err(EmitError::new(format!(
                "{stage} relation output function family @{symbol} gamma does not match operand"
            )));
        }
    }
    let eval_end = eval_start + evals.len();
    let eval_operands = operand_symbols(operation, eval_start, eval_end)?;
    if evals != eval_operands {
        return Err(EmitError::new(format!(
            "{stage} relation output function family @{symbol} evals do not match operands"
        )));
    }
    let factor_operands = operand_symbols(operation, eval_end, operation.operand_count())?;
    if factors != factor_operands {
        return Err(EmitError::new(format!(
            "{stage} relation output function family @{symbol} factors do not match operands"
        )));
    }
    let mut factor_offset = 0;
    let mut terms = Vec::with_capacity(term_gamma_power_offsets.len());
    for (((gamma_power_offset, function), eval), factor_count) in term_gamma_power_offsets
        .into_iter()
        .zip(term_functions)
        .zip(evals)
        .zip(term_factor_counts)
    {
        let factor_end = factor_offset + factor_count;
        terms.push(RelationOutputFunctionFamilyTermPlan {
            gamma_power_offset,
            function: RelationOutputFunctionKind::from_cpu_attr(&function).map_err(|error| {
                EmitError::new(format!(
                    "{stage} relation output function family @{symbol}: {error}"
                ))
            })?,
            eval,
            factors: factors[factor_offset..factor_end].to_vec(),
        });
        factor_offset = factor_end;
    }
    Ok(RelationOutputFunctionFamilyPlan {
        symbol,
        gamma: gamma.into_iter().next(),
        terms,
    })
}

pub fn lower_boolean_zero_function_family_output(
    stage: &str,
    relation: &str,
    relation_output_function_families: &mut Vec<RelationOutputFunctionFamilyPlan>,
    relation_output_asts: &mut [RelationOutputAst],
) -> Result<Vec<RelationOutputFieldExprPlan>, EmitError> {
    let Some(claim) = relation_output_asts
        .iter_mut()
        .find(|claim| claim.relation == relation)
    else {
        return Ok(Vec::new());
    };
    let family_symbol = claim.expected_output.clone();
    let family_index = relation_output_function_families
        .iter()
        .position(|family| family.symbol == family_symbol)
        .ok_or_else(|| {
            EmitError::new(format!(
                "{stage} relation output for @{relation} references missing function family @{family_symbol}"
            ))
        })?;
    let family = relation_output_function_families[family_index].clone();
    if family.terms.is_empty() {
        return Err(EmitError::new(format!(
            "{stage} relation output function family @{family_symbol} has no scalar terms"
        )));
    }

    let prefix = relation_output_family_prefix(&family.symbol);
    let mut rows = Vec::new();
    let mut term_symbols = Vec::with_capacity(family.terms.len());
    let mut gamma_powers = BTreeMap::new();
    for (term_index, term) in family.terms.iter().enumerate() {
        if term.function != RelationOutputFunctionKind::BooleanZero {
            return Err(EmitError::new(format!(
                "{stage} relation output function family @{family_symbol} has unsupported function"
            )));
        }
        let square = format!("{prefix}.term{term_index}.boolean_zero.square");
        let boolean_zero = format!("{prefix}.term{term_index}.boolean_zero");
        rows.push(relation_output_field_expr(
            square.clone(),
            "field.product",
            vec![term.eval.clone(), term.eval.clone()],
        ));
        rows.push(relation_output_field_expr(
            boolean_zero.clone(),
            "field.sub",
            vec![square, term.eval.clone()],
        ));
        let mut operands = Vec::new();
        if let Some(gamma) = &family.gamma {
            if term.gamma_power_offset > 0 {
                operands.push(eval_family_gamma_power(
                    term.gamma_power_offset,
                    gamma,
                    prefix,
                    &mut gamma_powers,
                    &mut rows,
                ));
            }
        } else if term.gamma_power_offset > 0 {
            return Err(EmitError::new(format!(
                "{stage} relation output function family @{family_symbol} has gamma power without gamma"
            )));
        }
        operands.push(boolean_zero);
        operands.extend(term.factors.iter().cloned());
        push_relation_output_product_term(
            prefix,
            term_index,
            operands,
            &mut term_symbols,
            &mut rows,
        );
    }
    let claim_expr = format!("{prefix}.claim_expr");
    rows.push(relation_output_field_expr(
        claim_expr.clone(),
        "field.sum",
        term_symbols,
    ));
    claim.expected_output = claim_expr;
    let _removed_family = relation_output_function_families.remove(family_index);
    Ok(rows)
}

pub fn lower_eval_family_output(
    stage: &str,
    relation: &str,
    relation_output_eval_families: &mut Vec<RelationOutputEvalFamilyPlan>,
    relation_output_asts: &mut [RelationOutputAst],
) -> Result<Vec<RelationOutputFieldExprPlan>, EmitError> {
    let Some(claim) = relation_output_asts
        .iter_mut()
        .find(|claim| claim.relation == relation)
    else {
        return Ok(Vec::new());
    };
    let family_symbol = claim.expected_output.clone();
    let family_index = relation_output_eval_families
        .iter()
        .position(|family| family.symbol == family_symbol)
        .ok_or_else(|| {
            EmitError::new(format!(
                "{stage} relation output for @{relation} references missing eval family @{family_symbol}"
            ))
        })?;
    let family = relation_output_eval_families[family_index].clone();
    for term in &family.item_terms {
        verify_count(
            "relation output eval family item factors",
            &family.symbol,
            family.evals.len(),
            term.factors.len(),
        )?;
    }

    let prefix = relation_output_family_prefix(&family.symbol);
    let mut rows = Vec::new();
    let mut term_symbols = Vec::new();
    let mut gamma_powers = BTreeMap::new();
    let mut term_index = 0usize;
    for (eval_index, eval_symbol) in family.evals.iter().enumerate() {
        let gamma_base = eval_index * family.power_stride;
        for &offset in &family.value_term_offsets {
            let gamma_power_offset = gamma_base + offset;
            let operands = eval_family_term_operands(
                eval_symbol,
                gamma_power_offset,
                None,
                &family.gamma,
                prefix,
                &mut gamma_powers,
                &mut rows,
            );
            push_relation_output_product_term(
                prefix,
                term_index,
                operands,
                &mut term_symbols,
                &mut rows,
            );
            term_index += 1;
        }
        for term in &family.shared_terms {
            let gamma_power_offset = gamma_base + term.gamma_power_offset;
            let operands = eval_family_term_operands(
                eval_symbol,
                gamma_power_offset,
                Some(&term.factor),
                &family.gamma,
                prefix,
                &mut gamma_powers,
                &mut rows,
            );
            push_relation_output_product_term(
                prefix,
                term_index,
                operands,
                &mut term_symbols,
                &mut rows,
            );
            term_index += 1;
        }
        for term in &family.item_terms {
            let gamma_power_offset = gamma_base + term.gamma_power_offset;
            let operands = eval_family_term_operands(
                eval_symbol,
                gamma_power_offset,
                Some(&term.factors[eval_index]),
                &family.gamma,
                prefix,
                &mut gamma_powers,
                &mut rows,
            );
            push_relation_output_product_term(
                prefix,
                term_index,
                operands,
                &mut term_symbols,
                &mut rows,
            );
            term_index += 1;
        }
    }
    if term_symbols.is_empty() {
        return Err(EmitError::new(format!(
            "{stage} relation output eval family @{family_symbol} has no scalar terms"
        )));
    }

    let claim_expr = format!("{prefix}.claim_expr");
    rows.push(relation_output_field_expr(
        claim_expr.clone(),
        "field.sum",
        term_symbols,
    ));
    claim.expected_output = claim_expr;
    let _removed_family = relation_output_eval_families.remove(family_index);
    Ok(rows)
}

pub fn lower_eval_family_output_to_weighted_sum(
    stage: &str,
    relation: &str,
    relation_output_eval_families: &mut Vec<RelationOutputEvalFamilyPlan>,
    relation_output_asts: &mut [RelationOutputAst],
) -> Result<Vec<RelationOutputFieldExprPlan>, EmitError> {
    let Some(claim) = relation_output_asts
        .iter_mut()
        .find(|claim| claim.relation == relation)
    else {
        return Ok(Vec::new());
    };
    let family_symbol = claim.expected_output.clone();
    let family_index = relation_output_eval_families
        .iter()
        .position(|family| family.symbol == family_symbol)
        .ok_or_else(|| {
            EmitError::new(format!(
                "{stage} relation output for @{relation} references missing eval family @{family_symbol}"
            ))
        })?;
    let family = relation_output_eval_families[family_index].clone();
    if family.evals.is_empty() {
        return Err(EmitError::new(format!(
            "{stage} relation output eval family @{family_symbol} has no eval terms"
        )));
    }
    let term_count =
        family.value_term_offsets.len() + family.shared_terms.len() + family.item_terms.len();
    if term_count == 0 {
        return Err(EmitError::new(format!(
            "{stage} relation output eval family @{family_symbol} has no scalar terms"
        )));
    }
    for term in &family.item_terms {
        verify_count(
            "relation output eval family item factors",
            &family.symbol,
            family.evals.len(),
            term.factors.len(),
        )?;
    }

    let prefix = relation_output_family_prefix(&family.symbol);
    let claim_expr = format!("{prefix}.claim_expr");
    let mut operands = Vec::with_capacity(
        1 + family.evals.len()
            + family.shared_terms.len()
            + family.evals.len() * family.item_terms.len(),
    );
    operands.push(family.gamma.clone());
    operands.extend(family.evals.iter().cloned());
    operands.extend(family.shared_terms.iter().map(|term| term.factor.clone()));
    operands.extend(
        family
            .item_terms
            .iter()
            .flat_map(|term| term.factors.iter().cloned()),
    );
    let formula = power_strided_weighted_sum_formula(
        family.evals.len(),
        family.power_stride,
        &family.value_term_offsets,
        &family
            .shared_terms
            .iter()
            .map(|term| term.gamma_power_offset)
            .collect::<Vec<_>>(),
        &family
            .item_terms
            .iter()
            .map(|term| term.gamma_power_offset)
            .collect::<Vec<_>>(),
    );
    claim.expected_output.clone_from(&claim_expr);
    let _removed_family = relation_output_eval_families.remove(family_index);
    Ok(vec![relation_output_field_expr(
        claim_expr, &formula, operands,
    )])
}

pub fn lower_product_family_output(
    stage: &str,
    relation: &str,
    relation_output_product_families: &mut Vec<RelationOutputProductFamilyPlan>,
    relation_output_asts: &mut [RelationOutputAst],
) -> Result<Vec<RelationOutputFieldExprPlan>, EmitError> {
    let Some(claim) = relation_output_asts
        .iter_mut()
        .find(|claim| claim.relation == relation)
    else {
        return Ok(Vec::new());
    };
    let family_symbol = claim.expected_output.clone();
    let family_index = relation_output_product_families
        .iter()
        .position(|family| family.symbol == family_symbol)
        .ok_or_else(|| {
            EmitError::new(format!(
                "{stage} relation output for @{relation} references missing product family @{family_symbol}"
            ))
        })?;
    let family = relation_output_product_families[family_index].clone();

    let prefix = relation_output_family_prefix(&family.symbol);
    let mut rows = Vec::new();
    let mut term_symbols = Vec::with_capacity(family.terms.len());
    let mut gamma_powers = BTreeMap::new();
    for (term_index, term) in family.terms.iter().enumerate() {
        if !term.eval_families.is_empty() {
            return Err(EmitError::new(format!(
                "{stage} relation output product family @{family_symbol} requires field-vector product support"
            )));
        }
        let mut operands = Vec::new();
        if let Some(gamma) = &family.gamma {
            if term.gamma_power_offset > 0 {
                operands.push(eval_family_gamma_power(
                    term.gamma_power_offset,
                    gamma,
                    prefix,
                    &mut gamma_powers,
                    &mut rows,
                ));
            }
        }
        operands.extend(term.evals.iter().cloned());
        operands.extend(term.factors.iter().cloned());
        if operands.is_empty() {
            return Err(EmitError::new(format!(
                "{stage} relation output product family @{family_symbol} has an empty scalar term"
            )));
        }
        push_relation_output_product_term(
            prefix,
            term_index,
            operands,
            &mut term_symbols,
            &mut rows,
        );
    }

    let claim_expr = format!("{prefix}.claim_expr");
    rows.push(relation_output_field_expr(
        claim_expr.clone(),
        "field.sum",
        term_symbols,
    ));
    claim.expected_output = claim_expr;
    let _removed_family = relation_output_product_families.remove(family_index);
    Ok(rows)
}

fn eval_family_term_operands(
    eval_symbol: &str,
    gamma_power_offset: usize,
    factor: Option<&String>,
    gamma: &str,
    prefix: &str,
    gamma_powers: &mut BTreeMap<usize, String>,
    rows: &mut Vec<RelationOutputFieldExprPlan>,
) -> Vec<String> {
    let mut operands = vec![eval_symbol.to_owned()];
    if gamma_power_offset > 0 {
        operands.push(eval_family_gamma_power(
            gamma_power_offset,
            gamma,
            prefix,
            gamma_powers,
            rows,
        ));
    }
    if let Some(factor) = factor {
        operands.push(factor.clone());
    }
    operands
}

fn eval_family_gamma_power(
    exponent: usize,
    gamma: &str,
    prefix: &str,
    gamma_powers: &mut BTreeMap<usize, String>,
    rows: &mut Vec<RelationOutputFieldExprPlan>,
) -> String {
    if let Some(symbol) = gamma_powers.get(&exponent) {
        return symbol.clone();
    }
    let symbol = format!("{prefix}.gamma_pow_{exponent}");
    rows.push(relation_output_field_expr(
        symbol.clone(),
        format!("field.pow:{exponent}"),
        vec![gamma.to_owned()],
    ));
    let _old = gamma_powers.insert(exponent, symbol.clone());
    symbol
}

fn push_relation_output_product_term(
    prefix: &str,
    term_index: usize,
    operands: Vec<String>,
    term_symbols: &mut Vec<String>,
    rows: &mut Vec<RelationOutputFieldExprPlan>,
) {
    let symbol = format!("{prefix}.term{term_index}");
    rows.push(relation_output_field_expr(
        symbol.clone(),
        "field.product",
        operands,
    ));
    term_symbols.push(symbol);
}

fn relation_output_family_prefix(symbol: &str) -> &str {
    symbol.strip_suffix(".family").unwrap_or(symbol)
}

fn relation_output_field_expr(
    symbol: String,
    formula: impl Into<String>,
    operands: Vec<String>,
) -> RelationOutputFieldExprPlan {
    RelationOutputFieldExprPlan {
        symbol,
        formula: formula.into(),
        operands,
    }
}

pub trait FieldExprDependencies {
    fn symbol(&self) -> &str;
    fn operands(&self) -> &[String];
}

pub fn resolve_relation_outputs<T>(
    stage: &str,
    relation_output_values: &[StructuredPolynomialEvalPlan],
    relation_output_eval_families: &[RelationOutputEvalFamilyPlan],
    relation_output_product_families: &[RelationOutputProductFamilyPlan],
    relation_output_function_families: &[RelationOutputFunctionFamilyPlan],
    field_exprs: &[T],
    claim_asts: Vec<RelationOutputAst>,
) -> Result<Vec<RelationOutputPlan>, EmitError>
where
    T: FieldExprDependencies,
{
    let relation_output_values_by_symbol: BTreeMap<_, _> = relation_output_values
        .iter()
        .enumerate()
        .map(|(index, value)| (value.symbol.as_str(), index))
        .collect();
    let relation_output_eval_families_by_symbol: BTreeMap<_, _> = relation_output_eval_families
        .iter()
        .map(|family| (family.symbol.as_str(), family))
        .collect();
    let relation_output_product_families_by_symbol: BTreeMap<_, _> =
        relation_output_product_families
            .iter()
            .map(|family| (family.symbol.as_str(), family))
            .collect();
    if let Some(family) = relation_output_function_families.first() {
        return Err(EmitError::new(format!(
            "{stage} relation output function family @{} must be lowered before resolving relation outputs",
            family.symbol
        )));
    }
    let field_exprs_by_symbol: BTreeMap<_, _> = field_exprs
        .iter()
        .map(|expr| (expr.symbol(), expr))
        .collect();
    claim_asts
        .into_iter()
        .map(|claim| {
            verify_count(
                "relation output polynomial_evals",
                &claim.relation,
                claim.polynomial_evals.len(),
                claim.polynomial_eval_operands.len(),
            )?;
            if claim.polynomial_evals != claim.polynomial_eval_operands {
                return Err(EmitError::new(format!(
                    "{stage} relation output for @{} polynomial_evals do not match operands",
                    claim.relation
                )));
            }
            for symbol in &claim.polynomial_evals {
                if !relation_output_values_by_symbol.contains_key(symbol.as_str()) {
                    return Err(EmitError::new(format!(
                        "{stage} relation output for @{} references missing output value @{symbol}",
                        claim.relation
                    )));
                }
            }
            let dependencies = output_dependency_closure(
                &relation_output_eval_families_by_symbol,
                &relation_output_product_families_by_symbol,
                &field_exprs_by_symbol,
                std::iter::once(claim.expected_output.as_str()),
            );
            if let Some(family) = dependencies.first_eval_family() {
                return Err(EmitError::new(format!(
                    "{stage} relation output for @{} depends on unlowered eval family @{family}",
                    claim.relation
                )));
            }
            if let Some(family) = dependencies.first_product_family() {
                return Err(EmitError::new(format!(
                    "{stage} relation output for @{} depends on unlowered product family @{family}",
                    claim.relation
                )));
            }
            Ok(RelationOutputPlan::new(
                claim.relation,
                claim.expected_output,
            ))
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum OutputDependencyNode {
    Scalar(String),
    FieldExpr(String),
    EvalFamily(String),
    ProductFamily(String),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum OutputFamilyDependency {
    Eval(String),
    Product(String),
}

#[derive(Default)]
struct OutputDependencyClosure {
    families: BTreeSet<OutputFamilyDependency>,
}

impl OutputDependencyClosure {
    fn first_eval_family(&self) -> Option<&str> {
        self.families.iter().find_map(|family| match family {
            OutputFamilyDependency::Eval(value) => Some(value.as_str()),
            OutputFamilyDependency::Product(_) => None,
        })
    }

    fn first_product_family(&self) -> Option<&str> {
        self.families.iter().find_map(|family| match family {
            OutputFamilyDependency::Eval(_) => None,
            OutputFamilyDependency::Product(value) => Some(value.as_str()),
        })
    }
}

fn output_dependency_closure<'a, T>(
    relation_output_eval_families_by_symbol: &BTreeMap<&str, &RelationOutputEvalFamilyPlan>,
    relation_output_product_families_by_symbol: &BTreeMap<&str, &RelationOutputProductFamilyPlan>,
    field_exprs_by_symbol: &BTreeMap<&str, &T>,
    roots: impl Iterator<Item = &'a str>,
) -> OutputDependencyClosure
where
    T: FieldExprDependencies,
{
    let mut visited = BTreeSet::new();
    let mut dependencies = OutputDependencyClosure::default();
    let mut stack = roots
        .map(|symbol| {
            output_dependency_node(
                relation_output_eval_families_by_symbol,
                relation_output_product_families_by_symbol,
                field_exprs_by_symbol,
                symbol,
            )
        })
        .collect::<Vec<_>>();
    while let Some(node) = stack.pop() {
        if !visited.insert(node.clone()) {
            continue;
        }
        match node {
            OutputDependencyNode::Scalar(_) => {}
            OutputDependencyNode::FieldExpr(symbol) => {
                let Some(expr) = field_exprs_by_symbol.get(symbol.as_str()) else {
                    continue;
                };
                stack.extend(expr.operands().iter().map(|operand| {
                    output_dependency_node(
                        relation_output_eval_families_by_symbol,
                        relation_output_product_families_by_symbol,
                        field_exprs_by_symbol,
                        operand,
                    )
                }));
            }
            OutputDependencyNode::EvalFamily(symbol) => {
                let Some(family) = relation_output_eval_families_by_symbol.get(symbol.as_str())
                else {
                    continue;
                };
                let _inserted = dependencies
                    .families
                    .insert(OutputFamilyDependency::Eval(family.symbol.clone()));
                stack.push(output_dependency_node(
                    relation_output_eval_families_by_symbol,
                    relation_output_product_families_by_symbol,
                    field_exprs_by_symbol,
                    &family.gamma,
                ));
                stack.extend(family.evals.iter().map(|eval| {
                    output_dependency_node(
                        relation_output_eval_families_by_symbol,
                        relation_output_product_families_by_symbol,
                        field_exprs_by_symbol,
                        eval,
                    )
                }));
                stack.extend(family.shared_terms.iter().map(|term| {
                    output_dependency_node(
                        relation_output_eval_families_by_symbol,
                        relation_output_product_families_by_symbol,
                        field_exprs_by_symbol,
                        &term.factor,
                    )
                }));
                stack.extend(family.item_terms.iter().flat_map(|term| {
                    term.factors.iter().map(|factor| {
                        output_dependency_node(
                            relation_output_eval_families_by_symbol,
                            relation_output_product_families_by_symbol,
                            field_exprs_by_symbol,
                            factor,
                        )
                    })
                }));
            }
            OutputDependencyNode::ProductFamily(symbol) => {
                let Some(family) = relation_output_product_families_by_symbol.get(symbol.as_str())
                else {
                    continue;
                };
                let _inserted = dependencies
                    .families
                    .insert(OutputFamilyDependency::Product(family.symbol.clone()));
                stack.extend(family.gamma.iter().map(|gamma| {
                    output_dependency_node(
                        relation_output_eval_families_by_symbol,
                        relation_output_product_families_by_symbol,
                        field_exprs_by_symbol,
                        gamma,
                    )
                }));
                for term in &family.terms {
                    stack.extend(term.evals.iter().map(|eval| {
                        output_dependency_node(
                            relation_output_eval_families_by_symbol,
                            relation_output_product_families_by_symbol,
                            field_exprs_by_symbol,
                            eval,
                        )
                    }));
                    stack.extend(term.factors.iter().map(|factor| {
                        output_dependency_node(
                            relation_output_eval_families_by_symbol,
                            relation_output_product_families_by_symbol,
                            field_exprs_by_symbol,
                            factor,
                        )
                    }));
                }
            }
        }
    }
    dependencies
}

fn output_dependency_node(
    relation_output_eval_families_by_symbol: &BTreeMap<&str, &RelationOutputEvalFamilyPlan>,
    relation_output_product_families_by_symbol: &BTreeMap<&str, &RelationOutputProductFamilyPlan>,
    field_exprs_by_symbol: &BTreeMap<&str, &impl FieldExprDependencies>,
    symbol: &str,
) -> OutputDependencyNode {
    if relation_output_eval_families_by_symbol.contains_key(symbol) {
        OutputDependencyNode::EvalFamily(symbol.to_owned())
    } else if relation_output_product_families_by_symbol.contains_key(symbol) {
        OutputDependencyNode::ProductFamily(symbol.to_owned())
    } else if field_exprs_by_symbol.contains_key(symbol) {
        OutputDependencyNode::FieldExpr(symbol.to_owned())
    } else {
        OutputDependencyNode::Scalar(symbol.to_owned())
    }
}

pub fn prune_output_only_field_exprs<'a, 'b, T>(
    field_exprs: &mut Vec<T>,
    sumcheck_claim_roots: impl Iterator<Item = &'a str>,
    relation_output_roots: impl Iterator<Item = &'b str>,
) where
    T: FieldExprDependencies,
{
    let field_exprs_by_symbol: BTreeMap<_, _> = field_exprs
        .iter()
        .map(|expr| (expr.symbol(), expr))
        .collect();
    let sumcheck_claim_closure =
        field_expr_dependency_closure(&field_exprs_by_symbol, sumcheck_claim_roots);
    let relation_output_closure =
        field_expr_dependency_closure(&field_exprs_by_symbol, relation_output_roots);
    field_exprs.retain(|expr| {
        !relation_output_closure.contains(expr.symbol())
            || sumcheck_claim_closure.contains(expr.symbol())
    });
}

pub struct RelationOutputVerification<'a> {
    pub relation_output_values: &'a [StructuredPolynomialEvalPlan],
    pub relation_output_eval_families: &'a [RelationOutputEvalFamilyPlan],
    pub relation_output_product_families: &'a [RelationOutputProductFamilyPlan],
    pub relation_output_function_families: &'a [RelationOutputFunctionFamilyPlan],
    pub relation_outputs: &'a [RelationOutputPlan],
    pub relations: &'a BTreeSet<String>,
    pub field_values: &'a VerifierScalarValueSet,
    pub point_values: &'a VerifierPointSourceSet,
}

pub fn verify_relation_outputs(
    stage: &str,
    verification: RelationOutputVerification<'_>,
) -> Result<(), EmitError> {
    let RelationOutputVerification {
        relation_output_values,
        relation_output_eval_families,
        relation_output_product_families,
        relation_output_function_families,
        relation_outputs,
        relations,
        field_values,
        point_values,
    } = verification;
    field_values.verify_no_conflicts(stage)?;
    point_values.verify_no_conflicts(stage)?;
    for polynomial_eval in relation_output_values {
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
    }
    for family in relation_output_eval_families {
        if !field_values.contains_symbol(&family.gamma) {
            return Err(EmitError::new(format!(
                "{stage} relation output eval family @{} references missing gamma @{}",
                family.symbol, family.gamma
            )));
        }
        for eval in &family.evals {
            if !field_values.contains_symbol(eval) {
                return Err(EmitError::new(format!(
                    "{stage} relation output eval family @{} references missing eval @{}",
                    family.symbol, eval
                )));
            }
        }
        for term in &family.shared_terms {
            if !field_values.contains_symbol(&term.factor) {
                return Err(EmitError::new(format!(
                    "{stage} relation output eval family @{} references missing shared factor @{}",
                    family.symbol, term.factor
                )));
            }
        }
        for term in &family.item_terms {
            verify_count(
                "relation output eval family item factors",
                &family.symbol,
                family.evals.len(),
                term.factors.len(),
            )?;
            for factor in &term.factors {
                if !field_values.contains_symbol(factor) {
                    return Err(EmitError::new(format!(
                        "{stage} relation output eval family @{} references missing item factor @{}",
                        family.symbol, factor
                    )));
                }
            }
        }
    }
    for family in relation_output_product_families {
        if let Some(gamma) = &family.gamma {
            if !field_values.contains_symbol(gamma) {
                return Err(EmitError::new(format!(
                    "{stage} relation output product family @{} references missing gamma @{}",
                    family.symbol, gamma
                )));
            }
        }
        for term in &family.terms {
            if term.evals.is_empty() && term.factors.is_empty() {
                return Err(EmitError::new(format!(
                    "{stage} relation output product family @{} has an empty term",
                    family.symbol
                )));
            }
            for eval in &term.evals {
                if !field_values.contains_symbol(eval) {
                    return Err(EmitError::new(format!(
                        "{stage} relation output product family @{} references missing eval @{}",
                        family.symbol, eval
                    )));
                }
            }
            for factor in &term.factors {
                if !field_values.contains_symbol(factor) {
                    return Err(EmitError::new(format!(
                        "{stage} relation output product family @{} references missing factor @{}",
                        family.symbol, factor
                    )));
                }
            }
        }
    }
    for family in relation_output_function_families {
        if let Some(gamma) = &family.gamma {
            if !field_values.contains_symbol(gamma) {
                return Err(EmitError::new(format!(
                    "{stage} relation output function family @{} references missing gamma @{}",
                    family.symbol, gamma
                )));
            }
        }
        for term in &family.terms {
            if !field_values.contains_symbol(&term.eval) {
                return Err(EmitError::new(format!(
                    "{stage} relation output function family @{} references missing eval @{}",
                    family.symbol, term.eval
                )));
            }
            for factor in &term.factors {
                if !field_values.contains_symbol(factor) {
                    return Err(EmitError::new(format!(
                        "{stage} relation output function family @{} references missing factor @{}",
                        family.symbol, factor
                    )));
                }
            }
        }
    }
    for claim in relation_outputs {
        if !relations.contains(&claim.relation) {
            return Err(EmitError::new(format!(
                "{stage} relation output references missing relation @{}",
                claim.relation
            )));
        }
        let expected_output = claim.expected_output_symbol();
        if !field_values.contains_ref(&claim.expected_output) {
            return Err(EmitError::new(format!(
                "{stage} relation output for @{} uses missing expected output @{}",
                claim.relation, expected_output
            )));
        }
        for local_scalar in &claim.local_scalars {
            if !field_values.contains_plan(local_scalar) {
                return Err(EmitError::new(format!(
                    "{stage} relation output for @{} references missing local scalar @{}",
                    claim.relation, local_scalar.symbol
                )));
            }
        }
    }
    Ok(())
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

fn verify_count(kind: &str, symbol: &str, expected: usize, actual: usize) -> Result<(), EmitError> {
    if expected == actual {
        Ok(())
    } else {
        Err(EmitError::new(format!(
            "{kind} @{symbol} count mismatch: expected {expected}, got {actual}"
        )))
    }
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::emit::rust::EmitError;
    use crate::protocols::jolt::verifier_values::{
        VerifierPointSourceKind, VerifierPointSourceSet, VerifierScalarValueKind,
        VerifierScalarValueSet,
    };

    use super::{
        resolve_relation_outputs, verify_relation_outputs, FieldExprDependencies,
        RelationOutputAst, RelationOutputEvalFamilyItemTermPlan, RelationOutputEvalFamilyPlan,
        RelationOutputEvalFamilySharedTermPlan, RelationOutputFunctionFamilyPlan,
        RelationOutputFunctionFamilyTermPlan, RelationOutputFunctionKind, RelationOutputPlan,
        RelationOutputProductFamilyPlan, RelationOutputProductFamilyTermPlan,
        RelationOutputVerification, StructuredPolynomialEvalPlan, StructuredPolynomialKind,
        StructuredPolynomialPointLength, StructuredPolynomialPointOrder,
        StructuredPolynomialPointPlan, StructuredPolynomialPointSegment,
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
    fn structured_polynomial_scalar_expr_plan_preserves_typed_point_transforms() {
        let value = StructuredPolynomialEvalPlan {
            symbol: "eq.output".to_owned(),
            polynomial: StructuredPolynomialKind::EqPlusOne,
            x_point: StructuredPolynomialPointPlan {
                source: "x.source".to_owned(),
                segment: StructuredPolynomialPointSegment::Prefix,
                length: StructuredPolynomialPointLength::YPoint,
                order: StructuredPolynomialPointOrder::Reverse,
            },
            y_point: StructuredPolynomialPointPlan {
                source: "y.source".to_owned(),
                segment: StructuredPolynomialPointSegment::Full,
                length: StructuredPolynomialPointLength::Full,
                order: StructuredPolynomialPointOrder::AsIs,
            },
        };

        let expr = super::structured_polynomial_scalar_expr_plan(&value);

        assert_eq!(expr.symbol, "eq.output");
        assert_eq!(
            expr.formula,
            "poly.structured_eval:eq_plus_one:prefix:y_point:reverse:full:full:as_is"
        );
        assert_eq!(
            expr.operands,
            vec!["x.source".to_owned(), "y.source".to_owned()]
        );
    }

    #[test]
    fn resolve_rejects_unlowered_eval_families_reachable_through_field_expressions(
    ) -> Result<(), EmitError> {
        let inner_family = RelationOutputEvalFamilyPlan {
            symbol: "inner.family".to_owned(),
            gamma: "inner.gamma".to_owned(),
            evals: vec!["inner.eval".to_owned()],
            power_stride: 1,
            value_term_offsets: vec![0],
            shared_terms: Vec::new(),
            item_terms: Vec::new(),
        };
        let outer_family = RelationOutputEvalFamilyPlan {
            symbol: "outer.family".to_owned(),
            gamma: "outer.gamma".to_owned(),
            evals: vec!["outer.eval".to_owned()],
            power_stride: 1,
            value_term_offsets: Vec::new(),
            shared_terms: vec![RelationOutputEvalFamilySharedTermPlan {
                gamma_power_offset: 0,
                factor: "outer.shared.factor".to_owned(),
            }],
            item_terms: vec![RelationOutputEvalFamilyItemTermPlan {
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
        let claim_asts = vec![RelationOutputAst {
            relation: "relation".to_owned(),
            polynomial_evals: Vec::new(),
            polynomial_eval_operands: Vec::new(),
            expected_output: "claim.expr".to_owned(),
        }];

        let error = match resolve_relation_outputs(
            "test",
            &[],
            &[inner_family, outer_family],
            &[],
            &[],
            &field_exprs,
            claim_asts,
        ) {
            Ok(_) => {
                return Err(EmitError::new(
                    "unlowered eval family should fail relation output resolution",
                ));
            }
            Err(error) => error,
        };
        assert!(error.to_string().contains(
            "test relation output for @relation depends on unlowered eval family @inner.family"
        ));
        Ok(())
    }

    #[test]
    fn resolve_rejects_unlowered_product_families_reachable_through_field_expressions(
    ) -> Result<(), EmitError> {
        let product_family = RelationOutputProductFamilyPlan {
            symbol: "product.family".to_owned(),
            gamma: Some("product.gamma".to_owned()),
            terms: vec![RelationOutputProductFamilyTermPlan {
                gamma_power_offset: 2,
                evals: vec!["product.eval".to_owned()],
                eval_families: Vec::new(),
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
        let claim_asts = vec![RelationOutputAst {
            relation: "relation".to_owned(),
            polynomial_evals: Vec::new(),
            polynomial_eval_operands: Vec::new(),
            expected_output: "claim.expr".to_owned(),
        }];

        let error = match resolve_relation_outputs(
            "test",
            &[],
            &[],
            &[product_family],
            &[],
            &field_exprs,
            claim_asts,
        ) {
            Ok(_) => {
                return Err(EmitError::new(
                    "unlowered product family should fail relation output resolution",
                ));
            }
            Err(error) => error,
        };
        assert!(error.to_string().contains(
            "test relation output for @relation depends on unlowered product family @product.family"
        ));
        Ok(())
    }

    #[test]
    fn resolve_rejects_unlowered_function_families() -> Result<(), EmitError> {
        let function_family = RelationOutputFunctionFamilyPlan {
            symbol: "function.family".to_owned(),
            gamma: Some("function.gamma".to_owned()),
            terms: vec![RelationOutputFunctionFamilyTermPlan {
                gamma_power_offset: 0,
                function: RelationOutputFunctionKind::BooleanZero,
                eval: "function.eval".to_owned(),
                factors: vec!["function.factor.expr".to_owned()],
            }],
        };
        let claim_asts = vec![RelationOutputAst {
            relation: "relation".to_owned(),
            polynomial_evals: Vec::new(),
            polynomial_eval_operands: Vec::new(),
            expected_output: "function.family".to_owned(),
        }];

        let error = match resolve_relation_outputs(
            "test",
            &[],
            &[],
            &[],
            &[function_family],
            &[] as &[TestFieldExpr],
            claim_asts,
        ) {
            Ok(_) => {
                return Err(EmitError::new(
                    "unlowered function family should fail relation output resolution",
                ));
            }
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("test relation output function family @function.family must be lowered"));
        Ok(())
    }

    #[test]
    fn relation_output_verification_rejects_conflicting_scalar_values() -> Result<(), EmitError> {
        let mut field_values = VerifierScalarValueSet::default();
        field_values.insert("value", VerifierScalarValueKind::OpeningInput);
        field_values.insert("value", VerifierScalarValueKind::FieldExpr);
        let point_values = VerifierPointSourceSet::default();
        let relations = BTreeSet::new();

        let error = match verify_relation_outputs(
            "stage",
            RelationOutputVerification {
                relation_output_values: &[],
                relation_output_eval_families: &[],
                relation_output_product_families: &[],
                relation_output_function_families: &[],
                relation_outputs: &[],
                relations: &relations,
                field_values: &field_values,
                point_values: &point_values,
            },
        ) {
            Ok(()) => {
                return Err(EmitError::new(
                    "conflicting scalar sources should fail verification",
                ));
            }
            Err(error) => error,
        };

        assert!(error.to_string().contains(
            "stage scalar value source @value has conflicting kinds OpeningInput and FieldExpr"
        ));
        Ok(())
    }

    #[test]
    fn relation_output_verification_rejects_conflicting_point_sources() -> Result<(), EmitError> {
        let field_values = VerifierScalarValueSet::default();
        let mut point_values = VerifierPointSourceSet::default();
        point_values.insert("point", VerifierPointSourceKind::OpeningInput);
        point_values.insert("point", VerifierPointSourceKind::PointExpr);
        let relations = BTreeSet::new();

        let error = match verify_relation_outputs(
            "stage",
            RelationOutputVerification {
                relation_output_values: &[],
                relation_output_eval_families: &[],
                relation_output_product_families: &[],
                relation_output_function_families: &[],
                relation_outputs: &[],
                relations: &relations,
                field_values: &field_values,
                point_values: &point_values,
            },
        ) {
            Ok(()) => {
                return Err(EmitError::new(
                    "conflicting point sources should fail verification",
                ));
            }
            Err(error) => error,
        };

        assert!(error.to_string().contains(
            "stage point source @point has conflicting kinds OpeningInput and PointExpr"
        ));
        Ok(())
    }

    #[test]
    fn relation_output_verification_requires_planned_local_scalars() -> Result<(), EmitError> {
        let mut field_values = VerifierScalarValueSet::default();
        field_values.insert("claim", VerifierScalarValueKind::FieldExpr);
        let point_values = VerifierPointSourceSet::default();
        let relations = BTreeSet::from(["relation".to_owned()]);
        let relation_outputs = [RelationOutputPlan::with_local_scalars(
            "relation",
            ["local.scalar".to_owned()],
            "claim",
        )];

        let error = match verify_relation_outputs(
            "stage",
            RelationOutputVerification {
                relation_output_values: &[],
                relation_output_eval_families: &[],
                relation_output_product_families: &[],
                relation_output_function_families: &[],
                relation_outputs: &relation_outputs,
                relations: &relations,
                field_values: &field_values,
                point_values: &point_values,
            },
        ) {
            Ok(()) => {
                return Err(EmitError::new(
                    "unplanned local scalar should fail relation output verification",
                ));
            }
            Err(error) => error,
        };

        assert!(error.to_string().contains(
            "stage relation output for @relation references missing local scalar @local.scalar"
        ));
        Ok(())
    }

    #[test]
    fn relation_output_verification_requires_relation_local_scalar_kind() -> Result<(), EmitError> {
        let mut field_values = VerifierScalarValueSet::default();
        field_values.insert("claim", VerifierScalarValueKind::FieldExpr);
        field_values.insert("local.scalar", VerifierScalarValueKind::FieldExpr);
        let point_values = VerifierPointSourceSet::default();
        let relations = BTreeSet::from(["relation".to_owned()]);
        let relation_outputs = [RelationOutputPlan::with_local_scalars(
            "relation",
            ["local.scalar".to_owned()],
            "claim",
        )];

        let error = match verify_relation_outputs(
            "stage",
            RelationOutputVerification {
                relation_output_values: &[],
                relation_output_eval_families: &[],
                relation_output_product_families: &[],
                relation_output_function_families: &[],
                relation_outputs: &relation_outputs,
                relations: &relations,
                field_values: &field_values,
                point_values: &point_values,
            },
        ) {
            Ok(()) => {
                return Err(EmitError::new(
                    "wrong local scalar value kind should fail relation output verification",
                ));
            }
            Err(error) => error,
        };

        assert!(error.to_string().contains(
            "stage relation output for @relation references missing local scalar @local.scalar"
        ));
        Ok(())
    }
}
