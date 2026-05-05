use melior::ir::operation::OperationRef;

use crate::ir::Cpu;

use super::super::super::support::{
    classify_compute_value_op, compute_to_cpu_op_name, ValueDialect, ValueOpFamily,
    CPU_FIELD_RESULT_TYPES, CPU_OPENING_INPUT_RESULT_TYPES, CPU_POINT_RESULT_TYPES,
};

pub(super) struct CpuValueDialect;

impl ValueDialect for CpuValueDialect {
    type Phase = Cpu;

    const OPENING_INPUT_RESULT_TYPES: &'static [&'static str] = CPU_OPENING_INPUT_RESULT_TYPES;
    const POINT_RESULT_TYPES: &'static [&'static str] = CPU_POINT_RESULT_TYPES;
    const FIELD_RESULT_TYPES: &'static [&'static str] = CPU_FIELD_RESULT_TYPES;

    fn classify(source_name: &str) -> Option<ValueOpFamily> {
        classify_compute_value_op(source_name)
    }

    fn target_op_name(operation: OperationRef<'_, '_>) -> String {
        compute_to_cpu_op_name(operation)
    }
}
