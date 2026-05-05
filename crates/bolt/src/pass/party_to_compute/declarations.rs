use melior::ir::operation::OperationRef;

use crate::ir::{BoltModule, Compute, Party};
use crate::mlir::{MeliorContext, MlirError};
use crate::schema::operation_name;

use super::super::support::{
    append_copied_named_op, compute_function_attrs, protocol_params_attrs, symbol_ref,
    COMPUTE_RELATION_ATTRS,
};
use super::PartyToComputeLowering;

pub(super) fn append_module_declarations<'c>(
    context: &'c MeliorContext,
    source_module: &BoltModule<'c, Party>,
    target_module: &BoltModule<'c, Compute>,
    options: PartyToComputeLowering<'_>,
) -> Result<(), MlirError> {
    let params_attrs = protocol_params_attrs(source_module)?;
    context.append_op_with_owned_attrs(
        target_module,
        "compute.params",
        Some(options.params_symbol),
        &params_attrs,
    )?;

    let source_symbol = symbol_ref(options.source_symbol);
    let function_attrs = compute_function_attrs(&source_symbol);
    context.append_op(
        target_module,
        "compute.function",
        Some(options.function_symbol),
        &function_attrs,
    )
}

pub(super) fn copy_relation_op<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Compute>,
    operation: OperationRef<'_, '_>,
) -> Result<bool, MlirError> {
    match operation_name(operation).as_str() {
        "piop.relation" => {
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
