use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Cpu};
use crate::mlir::{MeliorContext, MlirError};
use crate::schema::operation_name;

mod buffer;
mod family;

pub(super) fn lower_op<'c, 'a>(
    context: &'c MeliorContext,
    cpu: &'a BoltModule<'c, Cpu>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
) -> Result<bool, MlirError> {
    match operation_name(op).as_str() {
        "compute.oracle_dense_trace"
        | "compute.oracle_one_hot_chunk"
        | "compute.oracle_optional_advice"
        | "compute.oracle_ref" => {
            buffer::lower(context, cpu, value_map, op)?;
        }
        "compute.oracle_family_init" => {
            family::lower_init(context, cpu, value_map, op)?;
        }
        "compute.oracle_family_append" => {
            family::lower_append(context, cpu, value_map, op)?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}
