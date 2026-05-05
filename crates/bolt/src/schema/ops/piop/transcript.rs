use melior::ir::OperationRef;

use super::super::support::{
    attrs_shape, MaybeValidation, NO_OPERANDS_ONE_RESULT, ONE_OPERAND_ONE_RESULT,
    ONE_OPERAND_TWO_RESULTS, TWO_OPERANDS_ONE_RESULT,
};
use super::notation::{
    TRANSCRIPT_ABSORB_ATTRS, TRANSCRIPT_ABSORB_BYTES_ATTRS, TRANSCRIPT_SQUEEZE_ATTRS,
    TRANSCRIPT_STATE_ATTRS,
};

pub(super) fn validate_op(operation: OperationRef<'_, '_>, name: &str) -> MaybeValidation {
    let result = match name {
        "transcript.absorb" | "transcript.absorb_optional" => {
            attrs_shape(operation, TRANSCRIPT_ABSORB_ATTRS, TWO_OPERANDS_ONE_RESULT)
        }
        "transcript.absorb_bytes" => attrs_shape(
            operation,
            TRANSCRIPT_ABSORB_BYTES_ATTRS,
            ONE_OPERAND_ONE_RESULT,
        ),
        "transcript.squeeze" => {
            attrs_shape(operation, TRANSCRIPT_SQUEEZE_ATTRS, ONE_OPERAND_TWO_RESULTS)
        }
        "transcript.state" => {
            attrs_shape(operation, TRANSCRIPT_STATE_ATTRS, NO_OPERANDS_ONE_RESULT)
        }
        _ => return None,
    };
    Some(result)
}
