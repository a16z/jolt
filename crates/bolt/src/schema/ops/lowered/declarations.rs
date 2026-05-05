use melior::ir::OperationRef;

use super::super::support::{attrs, MaybeValidation};
use super::attrs::{FUNCTION_ATTRS, KERNEL_ATTRS, PARAMS_ATTRS, RELATION_ATTRS};
use super::dialect::LoweredDialect;

pub(super) fn validate_op<D: LoweredDialect>(
    operation: OperationRef<'_, '_>,
    suffix: &str,
) -> MaybeValidation {
    let result = match suffix {
        "params" => attrs(operation, PARAMS_ATTRS),
        "function" => attrs(operation, FUNCTION_ATTRS),
        "relation" if D::CAPABILITIES.has_relation_op() => attrs(operation, RELATION_ATTRS),
        "kernel" => attrs(operation, KERNEL_ATTRS),
        _ => return None,
    };
    Some(result)
}
