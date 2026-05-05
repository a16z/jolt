mod attrs;
mod operands;

pub(super) use attrs::{
    attr_error, bool_attr, i64_int_attr, int_array_attr, int_attr, signed_int_attr, string_attr,
    symbol_array_attr, symbol_attr, symbol_reference_attr,
};
pub(super) use operands::{operand_symbol, operand_symbols};

use melior::ir::operation::OperationLike;

pub(super) fn operation_name<'c: 'a, 'a>(operation: impl OperationLike<'c, 'a>) -> String {
    operation
        .name()
        .as_string_ref()
        .as_str()
        .unwrap_or("<invalid-operation-name>")
        .to_owned()
}
