use melior::ir::operation::OperationLike;
use melior::ir::{Attribute, OperationRef};

use crate::emit::rust::EmitError;
use crate::ir::{string_attribute_value, symbol_attribute_value};

use super::operation_name;

pub(in crate::protocols::jolt::emit::rust) fn string_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<String, EmitError> {
    operation
        .attribute(attr)
        .ok()
        .and_then(string_attribute_value)
        .ok_or_else(|| attr_error(operation, attr, "string"))
}

pub(in crate::protocols::jolt::emit::rust) fn symbol_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<String, EmitError> {
    operation
        .attribute(attr)
        .ok()
        .and_then(symbol_attribute_value)
        .ok_or_else(|| attr_error(operation, attr, "symbol"))
}

pub(in crate::protocols::jolt::emit::rust) fn symbol_reference_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<String, EmitError> {
    operation
        .attribute(attr)
        .ok()
        .and_then(symbol_attribute_value)
        .ok_or_else(|| attr_error(operation, attr, "symbol reference"))
}

pub(in crate::protocols::jolt::emit::rust) fn symbol_array_attr(
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

pub(in crate::protocols::jolt::emit::rust) fn int_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<usize, EmitError> {
    operation
        .attribute(attr)
        .map(parse_integer_attr)
        .ok()
        .flatten()
        .ok_or_else(|| attr_error(operation, attr, "integer"))
}

pub(in crate::protocols::jolt::emit::rust) fn i64_int_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<usize, EmitError> {
    let value = operation
        .attribute(attr)
        .ok()
        .and_then(|attribute| {
            attribute
                .to_string()
                .strip_suffix(" : i64")
                .map(str::to_owned)
        })
        .ok_or_else(|| attr_error(operation, attr, "integer"))?;
    value
        .parse()
        .map_err(|_| attr_error(operation, attr, "integer"))
}

pub(in crate::protocols::jolt::emit::rust) fn signed_int_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<isize, EmitError> {
    operation
        .attribute(attr)
        .map(parse_signed_integer_attr)
        .ok()
        .flatten()
        .ok_or_else(|| attr_error(operation, attr, "signed integer"))
}

pub(in crate::protocols::jolt::emit::rust) fn bool_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<bool, EmitError> {
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

pub(in crate::protocols::jolt::emit::rust) fn int_array_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<Vec<usize>, EmitError> {
    let attribute = operation
        .attribute(attr)
        .map(|attribute| attribute.to_string())
        .ok()
        .ok_or_else(|| attr_error(operation, attr, "integer array"))?;
    parse_int_array(&attribute).ok_or_else(|| attr_error(operation, attr, "integer array"))
}

pub(in crate::protocols::jolt::emit::rust) fn attr_error(
    operation: OperationRef<'_, '_>,
    attr: &str,
    expected: &str,
) -> EmitError {
    EmitError::new(format!(
        "{} attr `{attr}` is not a {expected}",
        operation_name(operation)
    ))
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

fn parse_integer_attr(attribute: Attribute<'_>) -> Option<usize> {
    attribute
        .to_string()
        .split_whitespace()
        .next()
        .and_then(|value| value.parse().ok())
}

fn parse_signed_integer_attr(attribute: Attribute<'_>) -> Option<isize> {
    attribute
        .to_string()
        .split_whitespace()
        .next()
        .and_then(|value| value.parse().ok())
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
