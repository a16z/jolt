use melior::ir::operation::{OperationLike, OperationRef, OperationResult};

use crate::ir::string_attribute_value;

use super::super::names::operation_name;
use crate::schema::SchemaError;

pub(in crate::schema) fn operand_owner_symbol(
    operation: OperationRef<'_, '_>,
    index: usize,
) -> Result<String, SchemaError> {
    let owner = operand_owner_result(operation, index)?;
    owner
        .owner()
        .attribute("sym_name")
        .ok()
        .and_then(string_attribute_value)
        .ok_or_else(|| {
            SchemaError::new(format!(
                "{} operand {index} owner missing sym_name",
                operation_name(operation)
            ))
        })
}

pub(in crate::schema) fn operand_owner_result<'c, 'a>(
    operation: OperationRef<'c, 'a>,
    index: usize,
) -> Result<OperationResult<'c, 'a>, SchemaError> {
    let operand = operation.operand(index).map_err(|_| {
        SchemaError::new(format!(
            "{} missing required operand {index}",
            operation_name(operation)
        ))
    })?;
    OperationResult::try_from(operand).map_err(|_| {
        SchemaError::new(format!(
            "{} operand {index} must be an op result",
            operation_name(operation)
        ))
    })
}
