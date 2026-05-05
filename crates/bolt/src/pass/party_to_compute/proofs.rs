use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Compute, Role};
use crate::mlir::{MeliorContext, MlirError};

mod opening;
mod pcs;
mod sumcheck;

pub(super) fn lower_proof_op<'c, 'a>(
    context: &'c MeliorContext,
    compute: &'a BoltModule<'c, Compute>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
    role: &Role,
) -> Result<(), MlirError> {
    if sumcheck::lower_op(context, compute, value_map, op, role)? {
        return Ok(());
    }
    if opening::lower_op(context, compute, value_map, op)? {
        return Ok(());
    }
    if pcs::lower_op(context, compute, value_map, op, role)? {
        return Ok(());
    }
    Ok(())
}
