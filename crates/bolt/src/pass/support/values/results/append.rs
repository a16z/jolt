use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Phase};
use crate::mlir::{MeliorContext, MlirError};

use super::super::super::result_count::LoweredResultCount;
use super::mapping::insert_result_mapping;

pub(in crate::pass) fn append_and_map_result_count<'c, 'a, P: Phase>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, P>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    source: OperationRef<'_, '_>,
    target_name: &str,
    symbol: &str,
    attrs: &[(String, String)],
    operands: &[Value<'c, 'a>],
    result_types: &[&str],
    result_count: LoweredResultCount,
) -> Result<(), MlirError> {
    let operation = context.append_typed_op_with_owned_attrs(
        module,
        target_name,
        Some(symbol),
        attrs,
        operands,
        result_types,
    )?;
    for index in 0..result_count.as_usize() {
        insert_result_mapping(value_map, source, operation, index, index)?;
    }
    Ok(())
}
