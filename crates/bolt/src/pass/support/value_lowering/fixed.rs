use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::BoltModule;
use crate::mlir::{MeliorContext, MlirError};

use super::super::lowering::append_lowered_result_count;
use super::super::result_count::LoweredResultCount;
use super::ValueDialect;

pub(super) fn lower_fixed_results<'c, 'a, D>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, D::Phase>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
    attrs: &[&str],
    result_types: &[&str],
    result_count: LoweredResultCount,
) -> Result<(), MlirError>
where
    D: ValueDialect,
{
    let target_name = D::target_op_name(op);
    append_lowered_result_count(
        context,
        module,
        value_map,
        op,
        0,
        &target_name,
        attrs,
        result_types,
        result_count,
    )
}
