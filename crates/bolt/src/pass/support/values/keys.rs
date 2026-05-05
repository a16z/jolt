use melior::ir::operation::{OperationLike, OperationRef, OperationResult};

use crate::mlir::MlirError;
use crate::schema::operation_name;

use super::super::attrs::string_attr;
use super::super::diagnostic::schema_error;

pub(super) fn operation_result_key_at(
    operation: OperationRef<'_, '_>,
    index: usize,
) -> Result<String, MlirError> {
    let result = operation.result(index).map_err(|_| {
        schema_error(format!(
            "{} requires result {index}",
            operation_name(operation)
        ))
    })?;
    result_key(operation, result.result_number()).map_err(|_| {
        schema_error(format!(
            "{} result {index} owner missing sym_name",
            operation_name(operation)
        ))
    })
}

fn result_key(operation: OperationRef<'_, '_>, result_number: usize) -> Result<String, MlirError> {
    let symbol = string_attr(operation, "sym_name")?;
    Ok(format!("{symbol}#{result_number}"))
}

pub(super) fn operand_key(
    operation: OperationRef<'_, '_>,
    index: usize,
) -> Result<String, MlirError> {
    let operand = operation.operand(index).map_err(|_| {
        schema_error(format!(
            "{} requires operand {index}",
            operation_name(operation)
        ))
    })?;
    let owner = OperationResult::try_from(operand).map_err(|_| {
        schema_error(format!(
            "{} operand {index} must be an op result",
            operation_name(operation)
        ))
    })?;
    result_key(owner.owner(), owner.result_number()).map_err(|_| {
        schema_error(format!(
            "{} operand {index} owner missing sym_name",
            operation_name(operation)
        ))
    })
}
