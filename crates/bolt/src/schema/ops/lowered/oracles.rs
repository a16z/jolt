use melior::ir::OperationRef;

use super::super::support::{
    attrs_shape, MaybeValidation, NO_OPERANDS_ONE_RESULT, TWO_OPERANDS_ONE_RESULT,
};
use super::attrs::{
    DENSE_TRACE_ATTRS, ONE_HOT_CHUNK_ATTRS, OPTIONAL_ADVICE_ATTRS, ORACLE_FAMILY_APPEND_ATTRS,
    ORACLE_FAMILY_INIT_ATTRS, ORACLE_REF_ATTRS,
};

pub(super) fn validate_op(operation: OperationRef<'_, '_>, suffix: &str) -> MaybeValidation {
    let result = match suffix {
        "oracle_dense_trace" => attrs_shape(operation, DENSE_TRACE_ATTRS, NO_OPERANDS_ONE_RESULT),
        "oracle_one_hot_chunk" => {
            attrs_shape(operation, ONE_HOT_CHUNK_ATTRS, NO_OPERANDS_ONE_RESULT)
        }
        "oracle_optional_advice" => {
            attrs_shape(operation, OPTIONAL_ADVICE_ATTRS, NO_OPERANDS_ONE_RESULT)
        }
        "oracle_ref" => attrs_shape(operation, ORACLE_REF_ATTRS, NO_OPERANDS_ONE_RESULT),
        "oracle_family_init" => {
            attrs_shape(operation, ORACLE_FAMILY_INIT_ATTRS, NO_OPERANDS_ONE_RESULT)
        }
        "oracle_family_append" => attrs_shape(
            operation,
            ORACLE_FAMILY_APPEND_ATTRS,
            TWO_OPERANDS_ONE_RESULT,
        ),
        _ => return None,
    };
    Some(result)
}
