use melior::ir::operation::OperationRef;

use crate::ir::Compute;
use crate::schema::operation_name;

use super::super::super::super::support::{
    classify_compute_opening_op, OpeningDialect, OpeningOpFamily,
    COMPUTE_OPENING_BATCH_RESULT_TYPES, COMPUTE_OPENING_CLAIM_RESULT_TYPES,
};

pub(super) struct KernelResolutionOpeningDialect;

impl OpeningDialect for KernelResolutionOpeningDialect {
    type Phase = Compute;

    const CLAIM_RESULT_TYPES: &'static [&'static str] = COMPUTE_OPENING_CLAIM_RESULT_TYPES;
    const BATCH_RESULT_TYPES: &'static [&'static str] = COMPUTE_OPENING_BATCH_RESULT_TYPES;

    fn classify(source_name: &str) -> Option<OpeningOpFamily> {
        classify_compute_opening_op(source_name)
    }

    fn target_op_name(operation: OperationRef<'_, '_>) -> String {
        operation_name(operation)
    }
}
