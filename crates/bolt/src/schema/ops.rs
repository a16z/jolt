use melior::ir::OperationRef;

use super::constraints::{is_bolt_dialect_op, operation_name};
use super::SchemaError;

mod foundation;
mod lowered;
mod pcs;
mod piop;
mod support;

pub(super) fn validate_verifier_lowering_op(
    operation: OperationRef<'_, '_>,
) -> Result<(), SchemaError> {
    let name = operation_name(operation);
    if lowered::is_verifier_forbidden(&name) {
        return Err(SchemaError::new(format!(
            "verifier lowering must use verifier-specific ops, got `{name}`"
        )));
    }
    Ok(())
}

pub(super) fn validate_op(operation: OperationRef<'_, '_>) -> Result<(), SchemaError> {
    let name = operation_name(operation);
    if let Some(result) = foundation::validate_op(operation, &name) {
        return result;
    }
    if let Some(result) = piop::validate_op(operation, &name) {
        return result;
    }
    if let Some(result) = lowered::validate_op::<lowered::ComputeDialect>(operation, &name) {
        return result;
    }
    if let Some(result) = lowered::validate_op::<lowered::CpuDialect>(operation, &name) {
        return result;
    }
    if let Some(result) = pcs::validate_op(operation, &name) {
        return result;
    }
    if is_bolt_dialect_op(&name) {
        return Err(SchemaError::new(format!(
            "unknown Bolt op `{name}` in schema verifier"
        )));
    }
    Ok(())
}
