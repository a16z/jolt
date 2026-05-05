use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Cpu};
use crate::mlir::{MeliorContext, MlirError};

use super::super::super::super::support::{
    classify_compute_sumcheck_value_op, compute_to_cpu_op_name, lower_sumcheck_value_op,
    SumcheckValueDialect, SumcheckValueFamily, CPU_SUMCHECK_EVAL_RESULT_TYPES,
    CPU_SUMCHECK_INSTANCE_RESULT_TYPES,
};

struct CpuSumcheckValueDialect;

impl SumcheckValueDialect for CpuSumcheckValueDialect {
    type Phase = Cpu;

    const EVAL_RESULT_TYPES: &'static [&'static str] = CPU_SUMCHECK_EVAL_RESULT_TYPES;
    const INSTANCE_RESULT_TYPES: &'static [&'static str] = CPU_SUMCHECK_INSTANCE_RESULT_TYPES;

    fn classify(source_name: &str) -> Option<SumcheckValueFamily> {
        classify_compute_sumcheck_value_op(source_name)
    }

    fn target_op_name(operation: OperationRef<'_, '_>) -> String {
        compute_to_cpu_op_name(operation)
    }
}

pub(super) fn lower_op<'c, 'a>(
    context: &'c MeliorContext,
    cpu: &'a BoltModule<'c, Cpu>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
) -> Result<bool, MlirError> {
    lower_sumcheck_value_op::<CpuSumcheckValueDialect>(context, cpu, value_map, op)
}
