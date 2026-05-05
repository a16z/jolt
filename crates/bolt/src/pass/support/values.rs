mod keys;
mod results;

use std::collections::BTreeMap;

use melior::ir::operation::{OperationLike, OperationRef};
use melior::ir::Value;

use crate::mlir::MlirError;
use crate::schema::operation_name;

use super::diagnostic::schema_error;
use keys::operand_key;
pub(in crate::pass) use results::append_and_map_result_count;

pub(in crate::pass) fn lowered_operands<'c, 'a>(
    operation: OperationRef<'_, '_>,
    value_map: &BTreeMap<String, Value<'c, 'a>>,
    start_index: usize,
) -> Result<Vec<Value<'c, 'a>>, MlirError> {
    (start_index..operation.operand_count())
        .map(|index| {
            let key = operand_key(operation, index)?;
            value_map.get(&key).copied().ok_or_else(|| {
                schema_error(format!(
                    "{} operand {index} was not lowered",
                    operation_name(operation)
                ))
            })
        })
        .collect()
}

pub(in crate::pass) fn required_lowered_operand<'c, 'a>(
    operation: OperationRef<'_, '_>,
    value_map: &BTreeMap<String, Value<'c, 'a>>,
    index: usize,
    missing_message: impl Into<String>,
) -> Result<Value<'c, 'a>, MlirError> {
    let key = operand_key(operation, index)?;
    let missing_message = missing_message.into();
    value_map
        .get(&key)
        .copied()
        .ok_or_else(|| schema_error(missing_message))
}
