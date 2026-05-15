use std::collections::{BTreeMap, BTreeSet};

use melior::ir::operation::{OperationLike, OperationResult};
use melior::ir::{Attribute, OperationRef};

use crate::emit::rust::EmitError;
use crate::ir::string_attribute_value;
use crate::schema::operation_name;

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
pub enum SumcheckOutputFunctionKind {
    BooleanZero,
}

impl SumcheckOutputFunctionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BooleanZero => "boolean_zero",
        }
    }

    pub fn from_cpu_attr(value: &str) -> Result<Self, EmitError> {
        match value {
            "boolean_zero" => Ok(Self::BooleanZero),
            _ => Err(EmitError::new(format!(
                "unsupported output function `{value}`"
            ))),
        }
    }
}

impl PartialEq<&str> for SumcheckOutputFunctionKind {
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
    pub gamma: Option<String>,
    pub terms: Vec<SumcheckOutputProductFamilyTermPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckOutputFunctionFamilyTermPlan {
    pub gamma_power_offset: usize,
    pub function: SumcheckOutputFunctionKind,
    pub eval: String,
    pub factors: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckOutputFunctionFamilyPlan {
    pub symbol: String,
    pub gamma: Option<String>,
    pub terms: Vec<SumcheckOutputFunctionFamilyTermPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckOutputClaimPlan {
    pub relation: String,
    pub polynomial_evals: Vec<StructuredPolynomialEvalPlan>,
    pub eval_families: Vec<SumcheckOutputEvalFamilyPlan>,
    pub product_families: Vec<SumcheckOutputProductFamilyPlan>,
    pub function_families: Vec<SumcheckOutputFunctionFamilyPlan>,
    pub local_scalars: Vec<String>,
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
    let gamma = optional_symbol_array_attr(operation, "gamma")?;
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
    let eval_start = gamma.len();
    if let Some(gamma_symbol) = gamma.first() {
        let gamma_operand = operand_symbol(operation, 0)?;
        if gamma_operand != *gamma_symbol {
            return Err(EmitError::new(format!(
                "{stage} output product family @{symbol} gamma does not match operand"
            )));
        }
    }
    let eval_end = eval_start + evals.len();
    let eval_operands = operand_symbols(operation, eval_start, eval_end)?;
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
        gamma: gamma.into_iter().next(),
        terms,
    })
}

pub fn parse_output_function_family_plan(
    stage: &str,
    operation: OperationRef<'_, '_>,
) -> Result<SumcheckOutputFunctionFamilyPlan, EmitError> {
    let symbol = string_attr(operation, "sym_name")?;
    let gamma = optional_symbol_array_attr(operation, "gamma")?;
    let evals = symbol_array_attr(operation, "evals")?;
    let factors = symbol_array_attr(operation, "factors")?;
    let term_gamma_power_offsets = int_array_attr(operation, "term_gamma_power_offsets")?;
    let term_functions = string_array_attr(operation, "term_functions")?;
    let term_factor_counts = int_array_attr(operation, "term_factor_counts")?;
    verify_count(
        "output function family term functions",
        &symbol,
        term_gamma_power_offsets.len(),
        term_functions.len(),
    )?;
    verify_count(
        "output function family term factor counts",
        &symbol,
        term_gamma_power_offsets.len(),
        term_factor_counts.len(),
    )?;
    verify_count(
        "output function family evals",
        &symbol,
        term_gamma_power_offsets.len(),
        evals.len(),
    )?;
    verify_count(
        "output function family factors",
        &symbol,
        term_factor_counts.iter().sum(),
        factors.len(),
    )?;
    let eval_start = gamma.len();
    if let Some(gamma_symbol) = gamma.first() {
        let gamma_operand = operand_symbol(operation, 0)?;
        if gamma_operand != *gamma_symbol {
            return Err(EmitError::new(format!(
                "{stage} output function family @{symbol} gamma does not match operand"
            )));
        }
    }
    let eval_end = eval_start + evals.len();
    let eval_operands = operand_symbols(operation, eval_start, eval_end)?;
    if evals != eval_operands {
        return Err(EmitError::new(format!(
            "{stage} output function family @{symbol} evals do not match operands"
        )));
    }
    let factor_operands = operand_symbols(operation, eval_end, operation.operand_count())?;
    if factors != factor_operands {
        return Err(EmitError::new(format!(
            "{stage} output function family @{symbol} factors do not match operands"
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
        terms.push(SumcheckOutputFunctionFamilyTermPlan {
            gamma_power_offset,
            function: SumcheckOutputFunctionKind::from_cpu_attr(&function).map_err(|error| {
                EmitError::new(format!("{stage} output function family @{symbol}: {error}"))
            })?,
            eval,
            factors: factors[factor_offset..factor_end].to_vec(),
        });
        factor_offset = factor_end;
    }
    Ok(SumcheckOutputFunctionFamilyPlan {
        symbol,
        gamma: gamma.into_iter().next(),
        terms,
    })
}

pub trait FieldExprDependencies {
    fn symbol(&self) -> &str;
    fn operands(&self) -> &[String];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifierScalarSourceKind {
    OpeningInput,
    FieldConstant,
    TranscriptScalar,
    FieldExpr,
    PointDerived,
    SumcheckEval,
    StructuredPolynomialEval,
    OutputEvalFamily,
    OutputProductFamily,
    OutputFunctionFamily,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VerifierScalarSourceSet {
    symbols: BTreeMap<String, VerifierScalarSourceKind>,
    conflicts: Vec<VerifierSourceConflict<VerifierScalarSourceKind>>,
}

impl VerifierScalarSourceSet {
    pub fn insert(&mut self, symbol: &str, kind: VerifierScalarSourceKind) {
        match self.symbols.entry(symbol.to_owned()) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                let _entry = entry.insert(kind);
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                let existing = *entry.get();
                if existing != kind {
                    self.conflicts.push(VerifierSourceConflict {
                        symbol: symbol.to_owned(),
                        existing,
                        incoming: kind,
                    });
                }
            }
        }
    }

    pub fn extend<'a>(
        &mut self,
        symbols: impl IntoIterator<Item = &'a String>,
        kind: VerifierScalarSourceKind,
    ) {
        for symbol in symbols {
            self.insert(symbol, kind);
        }
    }

    pub fn contains(&self, symbol: &str) -> bool {
        self.symbols.contains_key(symbol)
    }

    fn verify_no_conflicts(&self, stage: &str) -> Result<(), EmitError> {
        let Some(conflict) = self.conflicts.first() else {
            return Ok(());
        };
        Err(conflicting_source_error(stage, "scalar", conflict))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifierPointSourceKind {
    OpeningInput,
    SumcheckInstance,
    PointZero,
    PointSlice,
    PointConcat,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VerifierPointSourceSet {
    symbols: BTreeMap<String, VerifierPointSourceKind>,
    conflicts: Vec<VerifierSourceConflict<VerifierPointSourceKind>>,
}

impl VerifierPointSourceSet {
    pub fn insert(&mut self, symbol: &str, kind: VerifierPointSourceKind) {
        match self.symbols.entry(symbol.to_owned()) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                let _entry = entry.insert(kind);
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                let existing = *entry.get();
                if existing != kind {
                    self.conflicts.push(VerifierSourceConflict {
                        symbol: symbol.to_owned(),
                        existing,
                        incoming: kind,
                    });
                }
            }
        }
    }

    pub fn extend<'a>(
        &mut self,
        symbols: impl IntoIterator<Item = &'a String>,
        kind: VerifierPointSourceKind,
    ) {
        for symbol in symbols {
            self.insert(symbol, kind);
        }
    }

    pub fn contains(&self, symbol: &str) -> bool {
        self.symbols.contains_key(symbol)
    }

    fn verify_no_conflicts(&self, stage: &str) -> Result<(), EmitError> {
        let Some(conflict) = self.conflicts.first() else {
            return Ok(());
        };
        Err(conflicting_source_error(stage, "point", conflict))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VerifierSourceConflict<K> {
    symbol: String,
    existing: K,
    incoming: K,
}

fn conflicting_source_error<K>(
    stage: &str,
    value_kind: &str,
    conflict: &VerifierSourceConflict<K>,
) -> EmitError
where
    K: std::fmt::Debug,
{
    EmitError::new(format!(
        "{stage} {value_kind} source @{} has conflicting kinds {:?} and {:?}",
        conflict.symbol, conflict.existing, conflict.incoming
    ))
}

pub fn resolve_output_claims<T>(
    stage: &str,
    output_values: &[StructuredPolynomialEvalPlan],
    output_families: &[SumcheckOutputEvalFamilyPlan],
    output_product_families: &[SumcheckOutputProductFamilyPlan],
    output_function_families: &[SumcheckOutputFunctionFamilyPlan],
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
    let output_function_families_by_symbol: BTreeMap<_, _> = output_function_families
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
            let dependencies = output_dependency_closure(
                &output_families_by_symbol,
                &output_product_families_by_symbol,
                &output_function_families_by_symbol,
                &field_exprs_by_symbol,
                std::iter::once(claim.claim_value.as_str()),
            );
            let eval_families = output_families
                .iter()
                .filter(|family| dependencies.contains_eval_family(&family.symbol))
                .cloned()
                .collect();
            let product_families = output_product_families
                .iter()
                .filter(|family| dependencies.contains_product_family(&family.symbol))
                .cloned()
                .collect();
            let function_families = output_function_families
                .iter()
                .filter(|family| dependencies.contains_function_family(&family.symbol))
                .cloned()
                .collect();
            Ok(SumcheckOutputClaimPlan {
                relation: claim.relation,
                polynomial_evals,
                eval_families,
                product_families,
                function_families,
                local_scalars: Vec::new(),
                claim_value: claim.claim_value,
            })
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum OutputDependencyNode {
    Scalar(String),
    FieldExpr(String),
    EvalFamily(String),
    ProductFamily(String),
    FunctionFamily(String),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum OutputFamilyDependency {
    Eval(String),
    Product(String),
    Function(String),
}

#[derive(Default)]
struct OutputDependencyClosure {
    families: BTreeSet<OutputFamilyDependency>,
}

impl OutputDependencyClosure {
    fn contains_eval_family(&self, symbol: &str) -> bool {
        self.families
            .iter()
            .any(|family| matches!(family, OutputFamilyDependency::Eval(value) if value == symbol))
    }

    fn contains_product_family(&self, symbol: &str) -> bool {
        self.families.iter().any(
            |family| matches!(family, OutputFamilyDependency::Product(value) if value == symbol),
        )
    }

    fn contains_function_family(&self, symbol: &str) -> bool {
        self.families.iter().any(
            |family| matches!(family, OutputFamilyDependency::Function(value) if value == symbol),
        )
    }
}

fn output_dependency_closure<'a, T>(
    output_families_by_symbol: &BTreeMap<&str, &SumcheckOutputEvalFamilyPlan>,
    output_product_families_by_symbol: &BTreeMap<&str, &SumcheckOutputProductFamilyPlan>,
    output_function_families_by_symbol: &BTreeMap<&str, &SumcheckOutputFunctionFamilyPlan>,
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
                output_families_by_symbol,
                output_product_families_by_symbol,
                output_function_families_by_symbol,
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
                        output_families_by_symbol,
                        output_product_families_by_symbol,
                        output_function_families_by_symbol,
                        field_exprs_by_symbol,
                        operand,
                    )
                }));
            }
            OutputDependencyNode::EvalFamily(symbol) => {
                let Some(family) = output_families_by_symbol.get(symbol.as_str()) else {
                    continue;
                };
                let _inserted = dependencies
                    .families
                    .insert(OutputFamilyDependency::Eval(family.symbol.clone()));
                stack.push(output_dependency_node(
                    output_families_by_symbol,
                    output_product_families_by_symbol,
                    output_function_families_by_symbol,
                    field_exprs_by_symbol,
                    &family.gamma,
                ));
                stack.extend(family.evals.iter().map(|eval| {
                    output_dependency_node(
                        output_families_by_symbol,
                        output_product_families_by_symbol,
                        output_function_families_by_symbol,
                        field_exprs_by_symbol,
                        eval,
                    )
                }));
                stack.extend(family.shared_terms.iter().map(|term| {
                    output_dependency_node(
                        output_families_by_symbol,
                        output_product_families_by_symbol,
                        output_function_families_by_symbol,
                        field_exprs_by_symbol,
                        &term.factor,
                    )
                }));
                stack.extend(family.item_terms.iter().flat_map(|term| {
                    term.factors.iter().map(|factor| {
                        output_dependency_node(
                            output_families_by_symbol,
                            output_product_families_by_symbol,
                            output_function_families_by_symbol,
                            field_exprs_by_symbol,
                            factor,
                        )
                    })
                }));
            }
            OutputDependencyNode::ProductFamily(symbol) => {
                let Some(family) = output_product_families_by_symbol.get(symbol.as_str()) else {
                    continue;
                };
                let _inserted = dependencies
                    .families
                    .insert(OutputFamilyDependency::Product(family.symbol.clone()));
                stack.extend(family.gamma.iter().map(|gamma| {
                    output_dependency_node(
                        output_families_by_symbol,
                        output_product_families_by_symbol,
                        output_function_families_by_symbol,
                        field_exprs_by_symbol,
                        gamma,
                    )
                }));
                for term in &family.terms {
                    stack.extend(term.evals.iter().map(|eval| {
                        output_dependency_node(
                            output_families_by_symbol,
                            output_product_families_by_symbol,
                            output_function_families_by_symbol,
                            field_exprs_by_symbol,
                            eval,
                        )
                    }));
                    stack.extend(term.factors.iter().map(|factor| {
                        output_dependency_node(
                            output_families_by_symbol,
                            output_product_families_by_symbol,
                            output_function_families_by_symbol,
                            field_exprs_by_symbol,
                            factor,
                        )
                    }));
                }
            }
            OutputDependencyNode::FunctionFamily(symbol) => {
                let Some(family) = output_function_families_by_symbol.get(symbol.as_str()) else {
                    continue;
                };
                let _inserted = dependencies
                    .families
                    .insert(OutputFamilyDependency::Function(family.symbol.clone()));
                stack.extend(family.gamma.iter().map(|gamma| {
                    output_dependency_node(
                        output_families_by_symbol,
                        output_product_families_by_symbol,
                        output_function_families_by_symbol,
                        field_exprs_by_symbol,
                        gamma,
                    )
                }));
                for term in &family.terms {
                    stack.push(output_dependency_node(
                        output_families_by_symbol,
                        output_product_families_by_symbol,
                        output_function_families_by_symbol,
                        field_exprs_by_symbol,
                        &term.eval,
                    ));
                    stack.extend(term.factors.iter().map(|factor| {
                        output_dependency_node(
                            output_families_by_symbol,
                            output_product_families_by_symbol,
                            output_function_families_by_symbol,
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
    output_families_by_symbol: &BTreeMap<&str, &SumcheckOutputEvalFamilyPlan>,
    output_product_families_by_symbol: &BTreeMap<&str, &SumcheckOutputProductFamilyPlan>,
    output_function_families_by_symbol: &BTreeMap<&str, &SumcheckOutputFunctionFamilyPlan>,
    field_exprs_by_symbol: &BTreeMap<&str, &impl FieldExprDependencies>,
    symbol: &str,
) -> OutputDependencyNode {
    if output_families_by_symbol.contains_key(symbol) {
        OutputDependencyNode::EvalFamily(symbol.to_owned())
    } else if output_product_families_by_symbol.contains_key(symbol) {
        OutputDependencyNode::ProductFamily(symbol.to_owned())
    } else if output_function_families_by_symbol.contains_key(symbol) {
        OutputDependencyNode::FunctionFamily(symbol.to_owned())
    } else if field_exprs_by_symbol.contains_key(symbol) {
        OutputDependencyNode::FieldExpr(symbol.to_owned())
    } else {
        OutputDependencyNode::Scalar(symbol.to_owned())
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
    pub output_function_families: &'a [SumcheckOutputFunctionFamilyPlan],
    pub output_claims: &'a [SumcheckOutputClaimPlan],
    pub relations: &'a BTreeSet<String>,
    pub field_values: &'a VerifierScalarSourceSet,
    pub point_values: &'a VerifierPointSourceSet,
}

pub fn verify_output_claims(
    stage: &str,
    verification: OutputClaimVerification<'_>,
) -> Result<(), EmitError> {
    let OutputClaimVerification {
        output_values,
        output_families,
        output_product_families,
        output_function_families,
        output_claims,
        relations,
        field_values,
        point_values,
    } = verification;
    field_values.verify_no_conflicts(stage)?;
    point_values.verify_no_conflicts(stage)?;
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
        if let Some(gamma) = &family.gamma {
            if !field_values.contains(gamma) {
                return Err(EmitError::new(format!(
                    "{stage} output product family @{} references missing gamma @{}",
                    family.symbol, gamma
                )));
            }
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
    for family in output_function_families {
        if let Some(gamma) = &family.gamma {
            if !field_values.contains(gamma) {
                return Err(EmitError::new(format!(
                    "{stage} output function family @{} references missing gamma @{}",
                    family.symbol, gamma
                )));
            }
        }
        for term in &family.terms {
            if !field_values.contains(&term.eval) {
                return Err(EmitError::new(format!(
                    "{stage} output function family @{} references missing eval @{}",
                    family.symbol, term.eval
                )));
            }
            for factor in &term.factors {
                if !field_values.contains(factor) {
                    return Err(EmitError::new(format!(
                        "{stage} output function family @{} references missing factor @{}",
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

fn optional_symbol_array_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<Vec<String>, EmitError> {
    let values = symbol_array_attr(operation, attr)?;
    if values.len() <= 1 {
        Ok(values)
    } else {
        Err(EmitError::new(format!(
            "{} attr `{attr}` expected zero or one symbols, got {}",
            operation_name(operation),
            values.len()
        )))
    }
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

fn string_array_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<Vec<String>, EmitError> {
    let attribute = operation
        .attribute(attr)
        .map(|attribute| attribute.to_string())
        .ok()
        .ok_or_else(|| attr_error(operation, attr, "string array"))?;
    parse_string_array(&attribute).ok_or_else(|| attr_error(operation, attr, "string array"))
}

fn parse_string_array(attribute: &str) -> Option<Vec<String>> {
    let inner = attribute.strip_prefix('[')?.strip_suffix(']')?.trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    inner
        .split(',')
        .map(|item| {
            item.trim()
                .strip_prefix('"')?
                .strip_suffix('"')
                .map(ToOwned::to_owned)
        })
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
    use std::collections::BTreeSet;

    use crate::emit::rust::EmitError;

    use super::{
        resolve_output_claims, verify_output_claims, FieldExprDependencies,
        OutputClaimVerification, SumcheckOutputClaimAst, SumcheckOutputEvalFamilyItemTermPlan,
        SumcheckOutputEvalFamilyPlan, SumcheckOutputEvalFamilySharedTermPlan,
        SumcheckOutputFunctionFamilyPlan, SumcheckOutputFunctionFamilyTermPlan,
        SumcheckOutputFunctionKind, SumcheckOutputProductFamilyPlan,
        SumcheckOutputProductFamilyTermPlan, VerifierPointSourceKind, VerifierPointSourceSet,
        VerifierScalarSourceKind, VerifierScalarSourceSet,
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
        assert!(claim.function_families.is_empty());
        Ok(())
    }

    #[test]
    fn resolves_product_families_reachable_through_field_expressions() -> Result<(), EmitError> {
        let product_family = SumcheckOutputProductFamilyPlan {
            symbol: "product.family".to_owned(),
            gamma: Some("product.gamma".to_owned()),
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
            &[],
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
        assert!(claim.function_families.is_empty());
        Ok(())
    }

    #[test]
    fn resolves_function_families_reachable_through_field_expressions() -> Result<(), EmitError> {
        let function_family = SumcheckOutputFunctionFamilyPlan {
            symbol: "function.family".to_owned(),
            gamma: Some("function.gamma".to_owned()),
            terms: vec![SumcheckOutputFunctionFamilyTermPlan {
                gamma_power_offset: 0,
                function: SumcheckOutputFunctionKind::BooleanZero,
                eval: "function.eval".to_owned(),
                factors: vec!["function.factor.expr".to_owned()],
            }],
        };
        let field_exprs = vec![
            TestFieldExpr {
                symbol: "claim.expr".to_owned(),
                operands: vec!["function.family".to_owned()],
            },
            TestFieldExpr {
                symbol: "function.factor.expr".to_owned(),
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
            &[],
            &[function_family],
            &field_exprs,
            claim_asts,
        )?;
        let claim = claims
            .first()
            .ok_or_else(|| EmitError::new("missing resolved output claim"))?;
        assert!(claim.eval_families.is_empty());
        assert!(claim.product_families.is_empty());
        let family_symbols = claim
            .function_families
            .iter()
            .map(|family| family.symbol.as_str())
            .collect::<Vec<_>>();
        assert_eq!(family_symbols, vec!["function.family"]);
        Ok(())
    }

    #[test]
    fn output_claim_verification_rejects_conflicting_scalar_sources() -> Result<(), EmitError> {
        let mut field_values = VerifierScalarSourceSet::default();
        field_values.insert("value", VerifierScalarSourceKind::OpeningInput);
        field_values.insert("value", VerifierScalarSourceKind::FieldExpr);
        let point_values = VerifierPointSourceSet::default();
        let relations = BTreeSet::new();

        let error = match verify_output_claims(
            "stage",
            OutputClaimVerification {
                output_values: &[],
                output_families: &[],
                output_product_families: &[],
                output_function_families: &[],
                output_claims: &[],
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
            "stage scalar source @value has conflicting kinds OpeningInput and FieldExpr"
        ));
        Ok(())
    }

    #[test]
    fn output_claim_verification_rejects_conflicting_point_sources() -> Result<(), EmitError> {
        let field_values = VerifierScalarSourceSet::default();
        let mut point_values = VerifierPointSourceSet::default();
        point_values.insert("point", VerifierPointSourceKind::OpeningInput);
        point_values.insert("point", VerifierPointSourceKind::PointSlice);
        let relations = BTreeSet::new();

        let error = match verify_output_claims(
            "stage",
            OutputClaimVerification {
                output_values: &[],
                output_families: &[],
                output_product_families: &[],
                output_function_families: &[],
                output_claims: &[],
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
            "stage point source @point has conflicting kinds OpeningInput and PointSlice"
        ));
        Ok(())
    }
}
