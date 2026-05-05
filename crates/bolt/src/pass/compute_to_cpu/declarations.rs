use melior::ir::operation::OperationRef;

use crate::ir::{BoltModule, Cpu};
use crate::mlir::{MeliorContext, MlirError};
use crate::schema::operation_name;

use super::super::support::{
    append_copied_named_op, compute_params_symbol_ref_attrs, cpu_function_symbol_ref_attrs,
    string_attr, CPU_KERNEL_ATTRS,
};

pub(super) fn lower_declaration_op<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Cpu>,
    operation: OperationRef<'_, '_>,
) -> Result<bool, MlirError> {
    match operation_name(operation).as_str() {
        "compute.function" => {
            let symbol = string_attr(operation, "sym_name")?;
            let attrs = cpu_function_symbol_ref_attrs(operation)?;
            context.append_op_with_owned_attrs(module, "cpu.function", Some(&symbol), &attrs)?;
            Ok(true)
        }
        "compute.params" => {
            let symbol = string_attr(operation, "sym_name")?;
            let attrs = compute_params_symbol_ref_attrs(operation)?;
            context.append_op_with_owned_attrs(module, "cpu.params", Some(&symbol), &attrs)?;
            Ok(true)
        }
        "compute.kernel" => {
            append_copied_named_op(context, module, operation, "cpu.kernel", CPU_KERNEL_ATTRS)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}
