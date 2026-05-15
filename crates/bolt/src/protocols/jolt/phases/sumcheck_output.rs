use melior::ir::Value;

use crate::ir::{BoltModule, Protocol};
use crate::mlir::{MeliorContext, MlirError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StructuredPolynomialPointSpec<'a> {
    segment: &'a str,
    length: &'a str,
    order: &'a str,
}

impl<'a> StructuredPolynomialPointSpec<'a> {
    pub(crate) const fn full(order: &'a str) -> Self {
        Self {
            segment: "full",
            length: "full",
            order,
        }
    }

    pub(crate) const fn prefix(length: &'a str, order: &'a str) -> Self {
        Self {
            segment: "prefix",
            length,
            order,
        }
    }

    pub(crate) const fn suffix(length: &'a str, order: &'a str) -> Self {
        Self {
            segment: "suffix",
            length,
            order,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StructuredPolynomialSpec<'a> {
    pub(crate) symbol: &'a str,
    pub(crate) polynomial: &'a str,
    pub(crate) x_point: StructuredPolynomialPointSpec<'a>,
    pub(crate) y_point: StructuredPolynomialPointSpec<'a>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OutputClaimSpec<'a> {
    pub(crate) symbol: &'a str,
    pub(crate) stage: &'a str,
    pub(crate) relation: &'a str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OutputEvalFamilySpec<'a> {
    pub(crate) symbol: &'a str,
    pub(crate) power_stride: usize,
    pub(crate) value_term_offsets: &'a [usize],
    pub(crate) shared_term_offsets: &'a [usize],
    pub(crate) item_term_offsets: &'a [usize],
}

pub(crate) struct OutputProductFamilyTermSpec<'c, 'a> {
    pub(crate) gamma_power_offset: usize,
    pub(crate) evals: Vec<(String, Value<'c, 'a>)>,
    pub(crate) factors: Vec<(String, Value<'c, 'a>)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OutputProductFamilySpec<'a> {
    pub(crate) symbol: &'a str,
}

pub(crate) struct OutputFunctionFamilyTermSpec<'c, 'a> {
    pub(crate) gamma_power_offset: usize,
    pub(crate) function: &'static str,
    pub(crate) eval: (String, Value<'c, 'a>),
    pub(crate) factors: Vec<(String, Value<'c, 'a>)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OutputFunctionFamilySpec<'a> {
    pub(crate) symbol: &'a str,
}

pub(crate) fn append_structured_polynomial_eval<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    spec: StructuredPolynomialSpec<'_>,
    x_point: Value<'c, 'a>,
    y_point: Value<'c, 'a>,
) -> Result<Value<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "piop.structured_polynomial_eval",
        Some(spec.symbol),
        &[
            ("polynomial", &format!("\"{}\"", spec.polynomial)),
            ("x_point_segment", &format!("\"{}\"", spec.x_point.segment)),
            ("x_point_length", &format!("\"{}\"", spec.x_point.length)),
            ("x_point_order", &format!("\"{}\"", spec.x_point.order)),
            ("y_point_segment", &format!("\"{}\"", spec.y_point.segment)),
            ("y_point_length", &format!("\"{}\"", spec.y_point.length)),
            ("y_point_order", &format!("\"{}\"", spec.y_point.order)),
        ],
        &[x_point, y_point],
        &["!field.scalar"],
    )?;
    first_result(op, "piop.structured_polynomial_eval")
}

pub(crate) fn append_sumcheck_output_claim<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    spec: OutputClaimSpec<'_>,
    claim_value: Value<'c, 'a>,
    polynomial_evals: &[(&str, Value<'c, 'a>)],
) -> Result<(), MlirError> {
    let mut operands = Vec::with_capacity(polynomial_evals.len() + 1);
    operands.push(claim_value);
    operands.extend(polynomial_evals.iter().map(|(_, value)| *value));
    let local_value_symbols = polynomial_evals
        .iter()
        .map(|(symbol, _)| *symbol)
        .collect::<Vec<_>>();
    let _op = context.append_typed_op(
        module,
        "piop.sumcheck_output_claim",
        Some(spec.symbol),
        &[
            ("stage", &format!("@{}", spec.stage)),
            ("relation", &format!("@{}", spec.relation)),
            ("count", &int_attr(polynomial_evals.len())),
            ("polynomial_evals", &symbol_array_attr(&local_value_symbols)),
        ],
        &operands,
        &[],
    )?;
    Ok(())
}

pub(crate) fn append_sumcheck_output_eval_family<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    spec: OutputEvalFamilySpec<'_>,
    gamma: Value<'c, 'a>,
    evals: &[(&str, Value<'c, 'a>)],
    shared_terms: &[(&str, Value<'c, 'a>)],
    item_terms: &[(&str, Value<'c, 'a>)],
) -> Result<Value<'c, 'a>, MlirError> {
    let mut operands = Vec::with_capacity(1 + evals.len() + shared_terms.len() + item_terms.len());
    operands.push(gamma);
    operands.extend(evals.iter().map(|(_, value)| *value));
    operands.extend(shared_terms.iter().map(|(_, value)| *value));
    operands.extend(item_terms.iter().map(|(_, value)| *value));
    let eval_symbols = evals.iter().map(|(symbol, _)| *symbol).collect::<Vec<_>>();
    let shared_symbols = shared_terms
        .iter()
        .map(|(symbol, _)| *symbol)
        .collect::<Vec<_>>();
    let item_symbols = item_terms
        .iter()
        .map(|(symbol, _)| *symbol)
        .collect::<Vec<_>>();
    let op = context.append_typed_op(
        module,
        "piop.sumcheck_output_eval_family",
        Some(spec.symbol),
        &[
            ("power_stride", &int_attr(spec.power_stride)),
            (
                "value_term_offsets",
                &usize_array_attr(spec.value_term_offsets),
            ),
            (
                "shared_term_offsets",
                &usize_array_attr(spec.shared_term_offsets),
            ),
            (
                "item_term_offsets",
                &usize_array_attr(spec.item_term_offsets),
            ),
            ("evals", &symbol_array_attr(&eval_symbols)),
            ("shared_terms", &symbol_array_attr(&shared_symbols)),
            ("item_terms", &symbol_array_attr(&item_symbols)),
        ],
        &operands,
        &["!field.scalar"],
    )?;
    first_result(op, "piop.sumcheck_output_eval_family")
}

pub(crate) fn append_sumcheck_output_product_family<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    spec: OutputProductFamilySpec<'_>,
    gamma: Option<(&str, Value<'c, 'a>)>,
    terms: &[OutputProductFamilyTermSpec<'c, 'a>],
) -> Result<Value<'c, 'a>, MlirError> {
    let term_gamma_power_offsets = terms
        .iter()
        .map(|term| term.gamma_power_offset)
        .collect::<Vec<_>>();
    let term_eval_counts = terms
        .iter()
        .map(|term| term.evals.len())
        .collect::<Vec<_>>();
    let term_factor_counts = terms
        .iter()
        .map(|term| term.factors.len())
        .collect::<Vec<_>>();
    let mut operands = Vec::with_capacity(
        gamma.map_or(0, |_| 1)
            + term_eval_counts.iter().sum::<usize>()
            + term_factor_counts.iter().sum::<usize>(),
    );
    if let Some((_, gamma)) = gamma {
        operands.push(gamma);
    }
    operands.extend(
        terms
            .iter()
            .flat_map(|term| term.evals.iter().map(|(_, value)| *value)),
    );
    operands.extend(
        terms
            .iter()
            .flat_map(|term| term.factors.iter().map(|(_, value)| *value)),
    );
    let eval_symbols = terms
        .iter()
        .flat_map(|term| term.evals.iter().map(|(symbol, _)| symbol.as_str()))
        .collect::<Vec<_>>();
    let factor_symbols = terms
        .iter()
        .flat_map(|term| term.factors.iter().map(|(symbol, _)| symbol.as_str()))
        .collect::<Vec<_>>();
    let gamma_symbols = gamma.map(|(symbol, _)| vec![symbol]).unwrap_or_default();
    let op = context.append_typed_op(
        module,
        "piop.sumcheck_output_product_family",
        Some(spec.symbol),
        &[
            ("gamma", &symbol_array_attr(&gamma_symbols)),
            (
                "term_gamma_power_offsets",
                &usize_array_attr(&term_gamma_power_offsets),
            ),
            ("term_eval_counts", &usize_array_attr(&term_eval_counts)),
            ("term_factor_counts", &usize_array_attr(&term_factor_counts)),
            ("evals", &symbol_array_attr(&eval_symbols)),
            ("factors", &symbol_array_attr(&factor_symbols)),
        ],
        &operands,
        &["!field.scalar"],
    )?;
    first_result(op, "piop.sumcheck_output_product_family")
}

pub(crate) fn append_sumcheck_output_function_family<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    spec: OutputFunctionFamilySpec<'_>,
    gamma: Option<(&str, Value<'c, 'a>)>,
    terms: &[OutputFunctionFamilyTermSpec<'c, 'a>],
) -> Result<Value<'c, 'a>, MlirError> {
    let term_gamma_power_offsets = terms
        .iter()
        .map(|term| term.gamma_power_offset)
        .collect::<Vec<_>>();
    let term_factor_counts = terms
        .iter()
        .map(|term| term.factors.len())
        .collect::<Vec<_>>();
    let mut operands = Vec::with_capacity(
        gamma.map_or(0, |_| 1) + terms.len() + term_factor_counts.iter().sum::<usize>(),
    );
    if let Some((_, gamma)) = gamma {
        operands.push(gamma);
    }
    operands.extend(terms.iter().map(|term| term.eval.1));
    operands.extend(
        terms
            .iter()
            .flat_map(|term| term.factors.iter().map(|(_, value)| *value)),
    );
    let functions = terms.iter().map(|term| term.function).collect::<Vec<_>>();
    let eval_symbols = terms
        .iter()
        .map(|term| term.eval.0.as_str())
        .collect::<Vec<_>>();
    let factor_symbols = terms
        .iter()
        .flat_map(|term| term.factors.iter().map(|(symbol, _)| symbol.as_str()))
        .collect::<Vec<_>>();
    let gamma_symbols = gamma.map(|(symbol, _)| vec![symbol]).unwrap_or_default();
    let op = context.append_typed_op(
        module,
        "piop.sumcheck_output_function_family",
        Some(spec.symbol),
        &[
            ("gamma", &symbol_array_attr(&gamma_symbols)),
            (
                "term_gamma_power_offsets",
                &usize_array_attr(&term_gamma_power_offsets),
            ),
            ("term_functions", &string_array_attr(&functions)),
            ("term_factor_counts", &usize_array_attr(&term_factor_counts)),
            ("evals", &symbol_array_attr(&eval_symbols)),
            ("factors", &symbol_array_attr(&factor_symbols)),
        ],
        &operands,
        &["!field.scalar"],
    )?;
    first_result(op, "piop.sumcheck_output_function_family")
}

fn first_result<'c, 'a>(
    operation: melior::ir::operation::OperationRef<'c, 'a>,
    operation_name: &str,
) -> Result<Value<'c, 'a>, MlirError> {
    operation.result(0).map(Into::into).map_err(|_| {
        crate::schema::SchemaError::new(format!("{operation_name} requires result 0")).into()
    })
}

fn int_attr(value: usize) -> String {
    format!("{value} : i64")
}

fn usize_array_attr(values: &[usize]) -> String {
    let values = values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{values}]")
}

fn symbol_array_attr(values: &[&str]) -> String {
    let values = values
        .iter()
        .map(|value| format!("@{value}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{values}]")
}

fn string_array_attr(values: &[&str]) -> String {
    let values = values
        .iter()
        .map(|value| format!("\"{value}\""))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{values}]")
}
