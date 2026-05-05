use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::BoltModule;
use crate::mlir::{MeliorContext, MlirError};

use super::super::attrs::string_attr;
use super::super::result_count::LoweredResultCount;
use super::super::values::{append_and_map_result_count, lowered_operands};
use super::notation::field_expression_attrs;
use super::ValueDialect;

pub(super) fn lower_field_expression<'c, 'a, D>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, D::Phase>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
) -> Result<(), MlirError>
where
    D: ValueDialect,
{
    let target_name = D::target_op_name(op);
    let operands = lowered_operands(op, value_map, 0)?;
    let attrs = field_expression_attrs(op)?;
    let symbol = string_attr(op, "sym_name")?;
    append_and_map_result_count(
        context,
        module,
        value_map,
        op,
        &target_name,
        &symbol,
        &attrs,
        &operands,
        D::FIELD_RESULT_TYPES,
        LoweredResultCount::One,
    )
}
