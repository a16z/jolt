use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Cpu};
use crate::mlir::{MeliorContext, MlirError};

mod opening;
mod pcs;
mod sumcheck;

pub(super) fn lower_proof_op<'c, 'a>(
    context: &'c MeliorContext,
    cpu: &'a BoltModule<'c, Cpu>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
) -> Result<(), MlirError> {
    if sumcheck::lower_op(context, cpu, value_map, op)? {
        return Ok(());
    }
    if opening::lower_op(context, cpu, value_map, op)? {
        return Ok(());
    }
    if pcs::lower_op(context, cpu, value_map, op)? {
        return Ok(());
    }
    Ok(())
}
