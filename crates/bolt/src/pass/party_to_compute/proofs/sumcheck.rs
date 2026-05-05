mod targets;
mod values;

use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Compute, Role};
use crate::mlir::{MeliorContext, MlirError};
use crate::schema::operation_name;

use super::super::super::support::{
    lower_sumcheck_batch_op, lower_sumcheck_claim_op, lower_sumcheck_driver_op,
    COMPUTE_SUMCHECK_BATCH_RESULT_TYPES, COMPUTE_SUMCHECK_CLAIM_RESULT_TYPES,
    COMPUTE_SUMCHECK_DRIVER_RESULT_TYPES,
};
use targets::RoleSumcheckTargets;

pub(super) fn lower_op<'c, 'a>(
    context: &'c MeliorContext,
    compute: &'a BoltModule<'c, Compute>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
    role: &Role,
) -> Result<bool, MlirError> {
    let targets = RoleSumcheckTargets::for_role(role);
    if values::lower_op(context, compute, value_map, op)? {
        return Ok(true);
    }

    match operation_name(op).as_str() {
        "piop.sumcheck_claim" => {
            lower_sumcheck_claim_op(
                context,
                compute,
                value_map,
                op,
                targets.claim_op,
                COMPUTE_SUMCHECK_CLAIM_RESULT_TYPES,
            )?;
        }
        "piop.sumcheck_batch" => {
            lower_sumcheck_batch_op(
                context,
                compute,
                value_map,
                op,
                1,
                "compute.sumcheck_batch",
                COMPUTE_SUMCHECK_BATCH_RESULT_TYPES,
            )?;
        }
        "piop.sumcheck" => {
            lower_sumcheck_driver_op(
                context,
                compute,
                value_map,
                op,
                targets.driver_op,
                COMPUTE_SUMCHECK_DRIVER_RESULT_TYPES,
            )?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}
