use melior::ir::OperationRef;

use super::super::support::{
    attrs, attrs_shape, MaybeValidation, NO_OPERANDS_ONE_RESULT, ONE_OPERAND_NO_RESULTS,
};
use super::notation::{
    COMMIT_PUBLISH_BATCH_ATTRS, COMMIT_PUBLISH_OPTIONAL_ATTRS, ORACLE_ATTRS, ORACLE_FAMILY_ATTRS,
    PCS_COMMIT_BATCH_ATTRS,
};

pub(super) fn validate_op(operation: OperationRef<'_, '_>, name: &str) -> MaybeValidation {
    let result = match name {
        "piop.oracle" => attrs(operation, ORACLE_ATTRS),
        "piop.oracle_family" => attrs(operation, ORACLE_FAMILY_ATTRS),
        "commit.publish_batch" => attrs_shape(
            operation,
            COMMIT_PUBLISH_BATCH_ATTRS,
            NO_OPERANDS_ONE_RESULT,
        ),
        "commit.publish_optional" => attrs_shape(
            operation,
            COMMIT_PUBLISH_OPTIONAL_ATTRS,
            NO_OPERANDS_ONE_RESULT,
        ),
        "pcs.commit_batch" => {
            attrs_shape(operation, PCS_COMMIT_BATCH_ATTRS, ONE_OPERAND_NO_RESULTS)
        }
        _ => return None,
    };
    Some(result)
}
