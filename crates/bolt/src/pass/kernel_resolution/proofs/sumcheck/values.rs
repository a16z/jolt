use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Compute};
use crate::mlir::{MeliorContext, MlirError};
use crate::schema::operation_name;

use super::super::super::super::support::{
    classify_compute_sumcheck_value_op, lower_sumcheck_value_op, SumcheckValueDialect,
    SumcheckValueFamily, COMPUTE_SUMCHECK_EVAL_RESULT_TYPES,
    COMPUTE_SUMCHECK_INSTANCE_RESULT_TYPES,
};

struct KernelResolutionSumcheckValueDialect;

impl SumcheckValueDialect for KernelResolutionSumcheckValueDialect {
    type Phase = Compute;

    const EVAL_RESULT_TYPES: &'static [&'static str] = COMPUTE_SUMCHECK_EVAL_RESULT_TYPES;
    const INSTANCE_RESULT_TYPES: &'static [&'static str] = COMPUTE_SUMCHECK_INSTANCE_RESULT_TYPES;

    fn classify(source_name: &str) -> Option<SumcheckValueFamily> {
        classify_compute_sumcheck_value_op(source_name)
    }

    fn target_op_name(operation: OperationRef<'_, '_>) -> String {
        operation_name(operation)
    }
}

pub(super) fn lower_op<'c, 'a>(
    context: &'c MeliorContext,
    kernelized: &'a BoltModule<'c, Compute>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
) -> Result<bool, MlirError> {
    lower_sumcheck_value_op::<KernelResolutionSumcheckValueDialect>(
        context, kernelized, value_map, op,
    )
}
