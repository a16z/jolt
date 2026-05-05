use melior::ir::OperationRef;

use super::super::support::{
    attrs_shape, MaybeValidation, NO_OPERANDS_ONE_RESULT, ONE_OPERAND_ONE_RESULT,
    ONE_OPERAND_TWO_RESULTS, TWO_OPERANDS_ONE_RESULT,
};
use super::attrs::{
    TRANSCRIPT_ABSORB_ATTRS, TRANSCRIPT_ABSORB_BYTES_ATTRS, TRANSCRIPT_INIT_ATTRS,
    TRANSCRIPT_SQUEEZE_ATTRS,
};

pub(super) fn validate_op(operation: OperationRef<'_, '_>, suffix: &str) -> MaybeValidation {
    let result = match suffix {
        "transcript_init" => attrs_shape(operation, TRANSCRIPT_INIT_ATTRS, NO_OPERANDS_ONE_RESULT),
        "transcript_absorb" => {
            attrs_shape(operation, TRANSCRIPT_ABSORB_ATTRS, TWO_OPERANDS_ONE_RESULT)
        }
        "transcript_absorb_bytes" => attrs_shape(
            operation,
            TRANSCRIPT_ABSORB_BYTES_ATTRS,
            ONE_OPERAND_ONE_RESULT,
        ),
        "transcript_squeeze" => {
            attrs_shape(operation, TRANSCRIPT_SQUEEZE_ATTRS, ONE_OPERAND_TWO_RESULTS)
        }
        _ => return None,
    };
    Some(result)
}
