use melior::ir::operation::OperationRef;

use crate::ir::{BoltModule, Compute};
use crate::mlir::{MeliorContext, MlirError};
use crate::schema::operation_name;

use super::super::support::{
    append_copied_named_op, COMPUTE_FUNCTION_ATTRS, COMPUTE_RELATION_ATTRS, PROTOCOL_PARAM_ATTRS,
};

pub(super) fn copy_declaration_op<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Compute>,
    operation: OperationRef<'_, '_>,
) -> Result<bool, MlirError> {
    match operation_name(operation).as_str() {
        "compute.params" => {
            append_copied_named_op(
                context,
                module,
                operation,
                "compute.params",
                PROTOCOL_PARAM_ATTRS,
            )?;
            Ok(true)
        }
        "compute.function" => {
            append_copied_named_op(
                context,
                module,
                operation,
                "compute.function",
                COMPUTE_FUNCTION_ATTRS,
            )?;
            Ok(true)
        }
        "compute.relation" => {
            append_copied_named_op(
                context,
                module,
                operation,
                "compute.relation",
                COMPUTE_RELATION_ATTRS,
            )?;
            Ok(true)
        }
        _ => Ok(false),
    }
}
