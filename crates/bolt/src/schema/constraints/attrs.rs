mod parse;
mod readers;

use melior::ir::operation::{OperationLike, OperationRef};

use super::names::operation_name;
use crate::schema::SchemaError;

pub(in crate::schema) use readers::string_attr;
pub(crate) use readers::{int_attr, symbol_array_attr, symbol_attr};

pub(crate) fn require_attrs(
    operation: OperationRef<'_, '_>,
    attrs: &[&str],
) -> Result<(), SchemaError> {
    for attr in attrs {
        if !operation.has_attribute(attr) {
            return Err(SchemaError::new(format!(
                "{} missing required attr `{attr}`",
                operation_name(operation)
            )));
        }
    }
    Ok(())
}
