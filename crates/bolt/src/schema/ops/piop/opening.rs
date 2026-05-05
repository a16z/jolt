use melior::ir::OperationRef;

use super::super::support::{
    attrs_counted_min_shape, attrs_shape, opening_claim_equal, MaybeValidation,
    NO_OPERANDS_THREE_RESULTS, ORDERED_CLAIMS_WITH_NO_FIXED_OPERANDS, TWO_OPERANDS_ONE_RESULT,
};
use super::notation::{OPENING_BATCH_ATTRS, OPENING_CLAIM_ATTRS, OPENING_INPUT_ATTRS};

pub(super) fn validate_op(operation: OperationRef<'_, '_>, name: &str) -> MaybeValidation {
    let result = match name {
        "piop.opening_input" => {
            attrs_shape(operation, OPENING_INPUT_ATTRS, NO_OPERANDS_THREE_RESULTS)
        }
        "piop.opening_claim" => {
            attrs_shape(operation, OPENING_CLAIM_ATTRS, TWO_OPERANDS_ONE_RESULT)
        }
        "piop.opening_claim_equal" => opening_claim_equal(operation),
        "piop.opening_batch" => attrs_counted_min_shape(
            operation,
            OPENING_BATCH_ATTRS,
            ORDERED_CLAIMS_WITH_NO_FIXED_OPERANDS,
        ),
        _ => return None,
    };
    Some(result)
}
