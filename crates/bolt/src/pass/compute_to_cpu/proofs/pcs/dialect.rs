use melior::ir::operation::OperationRef;

use crate::ir::Cpu;
use crate::mlir::MlirError;

use super::super::super::super::support::{
    classify_compute_pcs_op, compute_to_cpu_op_name, PcsDialect, PcsLoweringRole, PcsOpFamily,
    CPU_OPENING_BATCH_OPENING_RESULT_TYPES, CPU_OPENING_BATCH_RESULT_TYPES,
    CPU_OPENING_CLAIM_RESULT_TYPES,
};

pub(super) struct CpuPcsDialect;

impl PcsDialect for CpuPcsDialect {
    type Phase = Cpu;

    const CLAIM_RESULT_TYPES: &'static [&'static str] = CPU_OPENING_CLAIM_RESULT_TYPES;
    const BATCH_RESULT_TYPES: &'static [&'static str] = CPU_OPENING_BATCH_RESULT_TYPES;
    const BATCH_OPENING_RESULT_TYPES: &'static [&'static str] =
        CPU_OPENING_BATCH_OPENING_RESULT_TYPES;

    fn classify(source_name: &str) -> Option<PcsOpFamily> {
        classify_compute_pcs_op(source_name)
    }

    fn target_op_name(
        operation: OperationRef<'_, '_>,
        _role: PcsLoweringRole<'_>,
    ) -> Result<String, MlirError> {
        Ok(compute_to_cpu_op_name(operation))
    }
}
