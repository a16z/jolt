use melior::ir::operation::OperationRef;

use crate::ir::Cpu;

use super::super::super::super::support::{
    classify_compute_opening_op, compute_to_cpu_op_name, OpeningDialect, OpeningOpFamily,
    CPU_OPENING_BATCH_RESULT_TYPES, CPU_OPENING_CLAIM_RESULT_TYPES,
};

pub(super) struct CpuOpeningDialect;

impl OpeningDialect for CpuOpeningDialect {
    type Phase = Cpu;

    const CLAIM_RESULT_TYPES: &'static [&'static str] = CPU_OPENING_CLAIM_RESULT_TYPES;
    const BATCH_RESULT_TYPES: &'static [&'static str] = CPU_OPENING_BATCH_RESULT_TYPES;

    fn classify(source_name: &str) -> Option<OpeningOpFamily> {
        classify_compute_opening_op(source_name)
    }

    fn target_op_name(operation: OperationRef<'_, '_>) -> String {
        compute_to_cpu_op_name(operation)
    }
}
