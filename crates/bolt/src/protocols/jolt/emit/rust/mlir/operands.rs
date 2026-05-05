use melior::ir::operation::{OperationLike, OperationResult};
use melior::ir::OperationRef;

use crate::emit::rust::EmitError;

use super::{operation_name, string_attr};

pub(in crate::protocols::jolt::emit::rust) fn operand_symbols(
    operation: OperationRef<'_, '_>,
    start_index: usize,
) -> Result<Vec<String>, EmitError> {
    (start_index..operation.operand_count())
        .map(|index| operand_symbol(operation, index))
        .collect()
}

pub(in crate::protocols::jolt::emit::rust) fn operand_symbol(
    operation: OperationRef<'_, '_>,
    index: usize,
) -> Result<String, EmitError> {
    let operand = operation.operand(index).map_err(|_| {
        EmitError::new(format!(
            "{} requires operand {index}",
            operation_name(operation)
        ))
    })?;
    let owner = OperationResult::try_from(operand).map_err(|_| {
        EmitError::new(format!(
            "{} operand {index} must be an op result",
            operation_name(operation)
        ))
    })?;
    string_attr(owner.owner(), "sym_name")
}
