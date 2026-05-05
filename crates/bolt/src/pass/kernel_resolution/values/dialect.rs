use melior::ir::operation::OperationRef;

use crate::ir::Compute;
use crate::schema::operation_name;

use super::super::super::support::{
    classify_compute_value_op, ValueDialect, ValueOpFamily, COMPUTE_FIELD_RESULT_TYPES,
    COMPUTE_OPENING_INPUT_RESULT_TYPES, COMPUTE_POINT_RESULT_TYPES,
};

pub(super) struct KernelResolutionValueDialect;

impl ValueDialect for KernelResolutionValueDialect {
    type Phase = Compute;

    const OPENING_INPUT_RESULT_TYPES: &'static [&'static str] = COMPUTE_OPENING_INPUT_RESULT_TYPES;
    const POINT_RESULT_TYPES: &'static [&'static str] = COMPUTE_POINT_RESULT_TYPES;
    const FIELD_RESULT_TYPES: &'static [&'static str] = COMPUTE_FIELD_RESULT_TYPES;

    fn classify(source_name: &str) -> Option<ValueOpFamily> {
        classify_compute_value_op(source_name)
    }

    fn target_op_name(operation: OperationRef<'_, '_>) -> String {
        operation_name(operation)
    }
}
