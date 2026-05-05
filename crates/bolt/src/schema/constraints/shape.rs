use melior::ir::operation::{OperationLike, OperationRef};

use super::names::operation_name;
use crate::schema::SchemaError;

mod counted;

pub(in crate::schema) use counted::require_counted_operands;

pub(in crate::schema) fn require_shape(
    operation: OperationRef<'_, '_>,
    operands: usize,
    results: usize,
) -> Result<(), SchemaError> {
    require_operand_count(operation, operands)?;
    require_result_count(operation, results)?;
    Ok(())
}

pub(in crate::schema) fn require_min_shape(
    operation: OperationRef<'_, '_>,
    min_operands: usize,
    results: usize,
) -> Result<(), SchemaError> {
    require_min_operand_count(operation, min_operands)?;
    require_result_count(operation, results)?;
    Ok(())
}

fn require_operand_count(
    operation: OperationRef<'_, '_>,
    operands: usize,
) -> Result<(), SchemaError> {
    if operation.operand_count() != operands {
        return Err(SchemaError::new(format!(
            "{} expected {operands} operands, got {}",
            operation_name(operation),
            operation.operand_count()
        )));
    }
    Ok(())
}

fn require_min_operand_count(
    operation: OperationRef<'_, '_>,
    min_operands: usize,
) -> Result<(), SchemaError> {
    if operation.operand_count() < min_operands {
        return Err(SchemaError::new(format!(
            "{} expected at least {min_operands} operands, got {}",
            operation_name(operation),
            operation.operand_count()
        )));
    }
    Ok(())
}

fn require_result_count(
    operation: OperationRef<'_, '_>,
    results: usize,
) -> Result<(), SchemaError> {
    if operation.result_count() != results {
        return Err(SchemaError::new(format!(
            "{} expected {results} results, got {}",
            operation_name(operation),
            operation.result_count()
        )));
    }
    Ok(())
}
