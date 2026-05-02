use melior::ir::operation::{OperationLike, OperationRef};

use crate::ir::string_attribute_value;
use crate::mlir::MlirError;
use crate::schema::{operation_name, SchemaError};

pub(super) fn copy_attrs(
    operation: OperationRef<'_, '_>,
    attrs: &[&str],
) -> Result<Vec<(String, String)>, MlirError> {
    attrs
        .iter()
        .filter_map(|attr| {
            operation
                .attribute(attr)
                .ok()
                .map(|value| Ok(((*attr).to_owned(), value.to_string())))
        })
        .collect()
}

pub(super) fn string_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<String, MlirError> {
    operation
        .attribute(attr)
        .ok()
        .and_then(string_attribute_value)
        .ok_or_else(|| {
            schema_error(format!(
                "{} attr `{attr}` is not a string",
                operation_name(operation)
            ))
        })
}

pub(super) fn transcript_squeeze_protocol_result_type(
    kind: &str,
) -> Result<&'static str, MlirError> {
    transcript_squeeze_value_type(kind, "!poly.point", "!field.scalar")
}

pub(super) fn transcript_squeeze_compute_result_types(
    operation: OperationRef<'_, '_>,
) -> Result<[&'static str; 2], MlirError> {
    Ok([
        "!compute.transcript_state",
        transcript_squeeze_value_type(
            string_attr(operation, "kind")?.as_str(),
            "!compute.point",
            "!compute.field_value",
        )?,
    ])
}

pub(super) fn transcript_squeeze_cpu_result_types(
    operation: OperationRef<'_, '_>,
) -> Result<[&'static str; 2], MlirError> {
    Ok([
        "!cpu.transcript_state",
        transcript_squeeze_value_type(
            string_attr(operation, "kind")?.as_str(),
            "!cpu.point",
            "!cpu.field_value",
        )?,
    ])
}

pub(super) fn field_lowering_attrs(
    operation: OperationRef<'_, '_>,
) -> Result<Vec<(String, String)>, MlirError> {
    match operation_name(operation).as_str() {
        "field.pow" | "compute.field_pow" => copy_attrs(operation, &["exponent"]),
        "poly.lagrange_basis_eval" | "compute.poly_lagrange_basis_eval" => {
            copy_attrs(operation, &["domain_start", "domain_size", "index"])
        }
        _ => Ok(Vec::new()),
    }
}

fn transcript_squeeze_value_type(
    kind: &str,
    point_type: &'static str,
    scalar_type: &'static str,
) -> Result<&'static str, MlirError> {
    match kind {
        "challenge_vector" => Ok(point_type),
        "challenge_scalar" | "scalar" => Ok(scalar_type),
        kind => Err(schema_error(format!(
            "unsupported transcript squeeze kind `{kind}`"
        ))),
    }
}

fn schema_error(message: impl Into<String>) -> MlirError {
    SchemaError::new(message).into()
}
