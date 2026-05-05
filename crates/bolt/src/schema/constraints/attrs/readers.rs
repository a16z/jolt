use melior::ir::operation::{OperationLike, OperationRef};

use crate::ir::{string_attribute_value, symbol_attribute_value};

use super::super::names::operation_name;
use super::parse::{parse_integer_attr, parse_symbol_array};
use crate::schema::SchemaError;

pub(crate) fn symbol_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<String, SchemaError> {
    operation
        .attribute(attr)
        .ok()
        .and_then(symbol_attribute_value)
        .ok_or_else(|| attr_error(operation, attr, "symbol"))
}

pub(in crate::schema) fn string_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<String, SchemaError> {
    operation
        .attribute(attr)
        .ok()
        .and_then(string_attribute_value)
        .ok_or_else(|| attr_error(operation, attr, "string"))
}

pub(crate) fn symbol_array_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<Vec<String>, SchemaError> {
    let attribute = operation
        .attribute(attr)
        .map(|attribute| attribute.to_string())
        .ok()
        .ok_or_else(|| attr_error(operation, attr, "symbol array"))?;
    parse_symbol_array(&attribute).ok_or_else(|| attr_error(operation, attr, "symbol array"))
}

pub(crate) fn int_attr(operation: OperationRef<'_, '_>, attr: &str) -> Result<usize, SchemaError> {
    operation
        .attribute(attr)
        .map(parse_integer_attr)
        .ok()
        .flatten()
        .ok_or_else(|| attr_error(operation, attr, "integer"))
}

fn attr_error(operation: OperationRef<'_, '_>, attr: &str, expected: &str) -> SchemaError {
    SchemaError::new(format!(
        "{} attr `{attr}` is not a {expected}",
        operation_name(operation)
    ))
}
