use melior::ir::operation::{OperationLike, OperationRef};

use crate::ir::{string_attribute_value, symbol_attribute_value};
use crate::mlir::MlirError;
use crate::schema::operation_name;

use super::super::diagnostic::schema_error;

pub(crate) fn string_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<String, MlirError> {
    operation
        .attribute(attr)
        .ok()
        .and_then(string_attribute_value)
        .ok_or_else(|| attr_error(operation, attr, "string"))
}

pub(in crate::pass) fn symbol_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<String, MlirError> {
    operation
        .attribute(attr)
        .ok()
        .and_then(symbol_attribute_value)
        .ok_or_else(|| attr_error(operation, attr, "symbol reference"))
}

pub(in crate::pass) fn bool_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<bool, MlirError> {
    operation
        .attribute(attr)
        .map(|attribute| match attribute.to_string().as_str() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        })
        .ok()
        .flatten()
        .ok_or_else(|| attr_error(operation, attr, "bool"))
}

fn attr_error(operation: OperationRef<'_, '_>, attr: &str, expected: &str) -> MlirError {
    schema_error(format!(
        "{} attr `{attr}` is not a {expected}",
        operation_name(operation)
    ))
}
