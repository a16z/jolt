mod attrs;
mod custom;
mod fixed;

use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Cpu};
use crate::mlir::{MeliorContext, MlirError};
use crate::schema::operation_name;

pub(super) fn lower_op<'c, 'a>(
    context: &'c MeliorContext,
    cpu: &'a BoltModule<'c, Cpu>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
) -> Result<bool, MlirError> {
    if fixed::lower_op(context, cpu, value_map, op)? {
        return Ok(true);
    }

    match operation_name(op).as_str() {
        "compute.transcript_init" => {
            custom::lower_init(context, cpu, value_map, op)?;
        }
        "compute.transcript_absorb" => {
            custom::lower_absorb(context, cpu, value_map, op)?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}
