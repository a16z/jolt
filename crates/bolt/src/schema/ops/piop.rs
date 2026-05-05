mod notation;
mod opening;
mod setup;
mod sumcheck;
mod transcript;

use melior::ir::OperationRef;

use super::support::{attrs, attrs_shape, MaybeValidation, NO_OPERANDS_ONE_RESULT};
use notation::RELATION_ATTRS;

pub(super) fn validate_op(operation: OperationRef<'_, '_>, name: &str) -> MaybeValidation {
    if let Some(result) = sumcheck::validate_op(operation, name) {
        return Some(result);
    }
    if let Some(result) = opening::validate_op(operation, name) {
        return Some(result);
    }
    if let Some(result) = transcript::validate_op(operation, name) {
        return Some(result);
    }
    if let Some(result) = setup::validate_op(operation, name) {
        return Some(result);
    }

    let result = match name {
        "piop.stage" => attrs_shape(
            operation,
            &["sym_name", "name", "order", "roles"],
            NO_OPERANDS_ONE_RESULT,
        ),
        "piop.relation" => attrs(operation, RELATION_ATTRS),
        _ => return None,
    };
    Some(result)
}
