use melior::ir::block::BlockLike;
use melior::ir::operation::{OperationLike, OperationRef};

use crate::ir::{string_attribute_value, BoltModule, Phase};

use super::attrs::symbol_attr;
use super::names::operation_name;
use crate::schema::SchemaError;

mod operands;

pub(in crate::schema) use operands::{operand_owner_result, operand_owner_symbol};

pub(crate) fn require_symbol_attr_eq(
    operation: OperationRef<'_, '_>,
    attr: &str,
    expected: &str,
) -> Result<(), SchemaError> {
    let actual = symbol_attr(operation, attr)?;
    if actual == expected {
        Ok(())
    } else {
        Err(SchemaError::new(format!(
            "{} attr `{attr}` expected @{expected}, got @{actual}",
            operation_name(operation)
        )))
    }
}

pub(crate) fn find_symbol<'c, P>(
    module: &'c BoltModule<'_, P>,
    symbol: &str,
) -> Option<OperationRef<'c, 'c>>
where
    P: Phase,
{
    let mut operation = module.as_mlir_module().body().first_operation();
    while let Some(op) = operation {
        operation = op.next_in_block();
        if op
            .attribute("sym_name")
            .ok()
            .and_then(string_attribute_value)
            .as_deref()
            == Some(symbol)
        {
            return Some(op);
        }
    }
    None
}
