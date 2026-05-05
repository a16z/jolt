use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Compute};
use crate::mlir::{MeliorContext, MlirError};

use super::registry::KernelRegistry;

mod opening;
mod pcs;
mod sumcheck;

pub(super) fn lower_proof_op<'c, 'a, R>(
    context: &'c MeliorContext,
    kernelized: &'a BoltModule<'c, Compute>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    kernels: &mut BTreeMap<String, String>,
    kernel_registry: &mut R,
    op: OperationRef<'_, '_>,
) -> Result<(), MlirError>
where
    R: KernelRegistry,
{
    if sumcheck::lower_op(context, kernelized, value_map, kernels, kernel_registry, op)? {
        return Ok(());
    }
    if opening::lower_op(context, kernelized, value_map, op)? {
        return Ok(());
    }
    if pcs::lower_op(context, kernelized, value_map, op)? {
        return Ok(());
    }
    Ok(())
}
