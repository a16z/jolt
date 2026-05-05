use melior::ir::operation::{OperationLike, OperationRef};

use super::super::attrs::{int_attr, symbol_array_attr};
use super::super::names::operation_name;
use super::super::symbols::operand_owner_symbol;
use crate::schema::SchemaError;

pub(in crate::schema) fn require_counted_operands(
    operation: OperationRef<'_, '_>,
    fixed_operands: usize,
    ordered_attr: &str,
) -> Result<(), SchemaError> {
    let count = int_attr(operation, "count")?;
    let dynamic_count = operation.operand_count().saturating_sub(fixed_operands);
    if count != dynamic_count {
        return Err(SchemaError::new(format!(
            "{} attr `count` expected {dynamic_count}, got {count}",
            operation_name(operation)
        )));
    }
    let ordered = symbol_array_attr(operation, ordered_attr)?;
    if ordered.len() != count {
        return Err(SchemaError::new(format!(
            "{} attr `{ordered_attr}` length {} does not match count {count}",
            operation_name(operation),
            ordered.len()
        )));
    }
    for (index, expected) in ordered.iter().enumerate() {
        let operand_index = fixed_operands + index;
        let actual = operand_owner_symbol(operation, operand_index)?;
        if &actual != expected {
            return Err(SchemaError::new(format!(
                "{} operand {operand_index} expected @{expected}, got @{actual}",
                operation_name(operation)
            )));
        }
    }
    Ok(())
}
