mod kernel;
mod values;

use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Compute};
use crate::mlir::{MeliorContext, MlirError};
use crate::schema::operation_name;

use super::super::super::support::{
    lower_sumcheck_batch_op, lower_sumcheck_claim_op, lower_sumcheck_driver_op,
    COMPUTE_SUMCHECK_BATCH_RESULT_TYPES, COMPUTE_SUMCHECK_CLAIM_RESULT_TYPES,
    COMPUTE_SUMCHECK_DRIVER_RESULT_TYPES,
};
use super::super::registry::KernelRegistry;

pub(super) fn lower_op<'c, 'a, R>(
    context: &'c MeliorContext,
    kernelized: &'a BoltModule<'c, Compute>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    kernels: &mut BTreeMap<String, String>,
    kernel_registry: &mut R,
    op: OperationRef<'_, '_>,
) -> Result<bool, MlirError>
where
    R: KernelRegistry,
{
    if values::lower_op(context, kernelized, value_map, op)? {
        return Ok(true);
    }

    match operation_name(op).as_str() {
        "compute.sumcheck_claim" => {
            kernel::lower_claim(context, kernelized, value_map, kernels, kernel_registry, op)?;
        }
        "compute.sumcheck_verify_claim" => {
            lower_sumcheck_claim_op(
                context,
                kernelized,
                value_map,
                op,
                "compute.sumcheck_verify_claim",
                COMPUTE_SUMCHECK_CLAIM_RESULT_TYPES,
            )?;
        }
        "compute.sumcheck_batch" => {
            lower_sumcheck_batch_op(
                context,
                kernelized,
                value_map,
                op,
                0,
                "compute.sumcheck_batch",
                COMPUTE_SUMCHECK_BATCH_RESULT_TYPES,
            )?;
        }
        "compute.sumcheck_driver" => {
            kernel::lower_driver(context, kernelized, value_map, kernels, kernel_registry, op)?;
        }
        "compute.sumcheck_verify" => {
            lower_sumcheck_driver_op(
                context,
                kernelized,
                value_map,
                op,
                "compute.sumcheck_verify",
                COMPUTE_SUMCHECK_DRIVER_RESULT_TYPES,
            )?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}
