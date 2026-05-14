use melior::ir::Value;

use crate::ir::{BoltModule, Protocol};
use crate::mlir::{MeliorContext, MlirError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OutputPointSpec<'a> {
    segment: &'a str,
    length: &'a str,
    order: &'a str,
}

impl<'a> OutputPointSpec<'a> {
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
pub(crate) struct OutputValueSpec<'a> {
    pub(crate) symbol: &'a str,
    pub(crate) kind: &'a str,
    pub(crate) local_point: OutputPointSpec<'a>,
    pub(crate) opening_point: OutputPointSpec<'a>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OutputClaimSpec<'a> {
    pub(crate) symbol: &'a str,
    pub(crate) stage: &'a str,
    pub(crate) relation: &'a str,
}

pub(crate) fn append_sumcheck_output_value<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    spec: OutputValueSpec<'_>,
    local_point: Value<'c, 'a>,
    opening_point: Value<'c, 'a>,
) -> Result<Value<'c, 'a>, MlirError> {
    let op = context.append_typed_op(
        module,
        "piop.sumcheck_output_value",
        Some(spec.symbol),
        &[
            ("kind", &format!("\"{}\"", spec.kind)),
            (
                "local_point_segment",
                &format!("\"{}\"", spec.local_point.segment),
            ),
            (
                "local_point_length",
                &format!("\"{}\"", spec.local_point.length),
            ),
            (
                "local_point_order",
                &format!("\"{}\"", spec.local_point.order),
            ),
            (
                "opening_point_segment",
                &format!("\"{}\"", spec.opening_point.segment),
            ),
            (
                "opening_point_length",
                &format!("\"{}\"", spec.opening_point.length),
            ),
            (
                "opening_point_order",
                &format!("\"{}\"", spec.opening_point.order),
            ),
        ],
        &[local_point, opening_point],
        &["!field.scalar"],
    )?;
    first_result(op, "piop.sumcheck_output_value")
}

pub(crate) fn append_sumcheck_output_claim<'c, 'a>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    spec: OutputClaimSpec<'_>,
    claim_value: Value<'c, 'a>,
    local_values: &[(&str, Value<'c, 'a>)],
) -> Result<(), MlirError> {
    let mut operands = Vec::with_capacity(local_values.len() + 1);
    operands.push(claim_value);
    operands.extend(local_values.iter().map(|(_, value)| *value));
    let local_value_symbols = local_values
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
            ("count", &int_attr(local_values.len())),
            ("local_values", &symbol_array_attr(&local_value_symbols)),
        ],
        &operands,
        &[],
    )?;
    Ok(())
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

fn symbol_array_attr(values: &[&str]) -> String {
    let values = values
        .iter()
        .map(|value| format!("@{value}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{values}]")
}
