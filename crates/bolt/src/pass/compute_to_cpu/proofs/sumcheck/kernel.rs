use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Cpu};
use crate::mlir::{MeliorContext, MlirError};

use crate::pass::support::{
    lower_kernel_sumcheck_claim_op, lower_kernel_sumcheck_driver_op,
    CPU_SUMCHECK_CLAIM_RESULT_TYPES, CPU_SUMCHECK_DRIVER_RESULT_TYPES,
};

pub(super) fn lower_claim<'c, 'a>(
    context: &'c MeliorContext,
    cpu: &'a BoltModule<'c, Cpu>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
) -> Result<(), MlirError> {
    lower_kernel_sumcheck_claim_op(
        context,
        cpu,
        value_map,
        op,
        "cpu.sumcheck_claim",
        CPU_SUMCHECK_CLAIM_RESULT_TYPES,
    )
}

pub(super) fn lower_driver<'c, 'a>(
    context: &'c MeliorContext,
    cpu: &'a BoltModule<'c, Cpu>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
) -> Result<(), MlirError> {
    lower_kernel_sumcheck_driver_op(
        context,
        cpu,
        value_map,
        op,
        "cpu.sumcheck_driver",
        CPU_SUMCHECK_DRIVER_RESULT_TYPES,
    )
}
