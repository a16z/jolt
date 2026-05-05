mod kernel;
mod values;

use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Cpu};
use crate::mlir::{MeliorContext, MlirError};
use crate::schema::operation_name;

use super::super::super::support::{
    lower_sumcheck_batch_op, lower_sumcheck_claim_op, lower_sumcheck_driver_op,
    CPU_SUMCHECK_BATCH_RESULT_TYPES, CPU_SUMCHECK_CLAIM_RESULT_TYPES,
    CPU_SUMCHECK_DRIVER_RESULT_TYPES,
};

pub(super) fn lower_op<'c, 'a>(
    context: &'c MeliorContext,
    cpu: &'a BoltModule<'c, Cpu>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
) -> Result<bool, MlirError> {
    if values::lower_op(context, cpu, value_map, op)? {
        return Ok(true);
    }

    match operation_name(op).as_str() {
        "compute.sumcheck_kernel_claim" => {
            kernel::lower_claim(context, cpu, value_map, op)?;
        }
        "compute.sumcheck_verify_claim" => {
            lower_sumcheck_claim_op(
                context,
                cpu,
                value_map,
                op,
                "cpu.sumcheck_verify_claim",
                CPU_SUMCHECK_CLAIM_RESULT_TYPES,
            )?;
        }
        "compute.sumcheck_batch" => {
            lower_sumcheck_batch_op(
                context,
                cpu,
                value_map,
                op,
                0,
                "cpu.sumcheck_batch",
                CPU_SUMCHECK_BATCH_RESULT_TYPES,
            )?;
        }
        "compute.sumcheck_kernel_driver" => {
            kernel::lower_driver(context, cpu, value_map, op)?;
        }
        "compute.sumcheck_verify" => {
            lower_sumcheck_driver_op(
                context,
                cpu,
                value_map,
                op,
                "cpu.sumcheck_verify",
                CPU_SUMCHECK_DRIVER_RESULT_TYPES,
            )?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}
