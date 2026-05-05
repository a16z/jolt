use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Phase};
use crate::mlir::{MeliorContext, MlirError};

use super::attrs::{copy_attrs, string_attr};
use super::result_count::LoweredResultCount;
use super::values::{append_and_map_result_count, lowered_operands};

pub(in crate::pass) fn append_copied_named_op<'c, P: Phase>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, P>,
    source: OperationRef<'_, '_>,
    target_name: &str,
    attr_names: &[&str],
) -> Result<(), MlirError> {
    let attrs = copy_attrs(source, attr_names)?;
    let symbol = string_attr(source, "sym_name")?;
    context.append_op_with_owned_attrs(module, target_name, Some(&symbol), &attrs)
}

pub(in crate::pass) fn append_lowered_result_count<'c, 'a, P: Phase>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, P>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    source: OperationRef<'_, '_>,
    operand_start: usize,
    target_name: &str,
    attr_names: &[&str],
    result_types: &[&str],
    result_count: LoweredResultCount,
) -> Result<(), MlirError> {
    let operands = lowered_operands(source, value_map, operand_start)?;
    let symbol = string_attr(source, "sym_name")?;
    let attrs = copy_attrs(source, attr_names)?;
    append_and_map_result_count(
        context,
        module,
        value_map,
        source,
        target_name,
        &symbol,
        &attrs,
        &operands,
        result_types,
        result_count,
    )
}
