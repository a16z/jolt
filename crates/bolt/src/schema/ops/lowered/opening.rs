use melior::ir::OperationRef;

use super::super::support::{
    attrs_counted_min_shape, attrs_shape, opening_claim_equal, MaybeValidation,
    ORDERED_CLAIMS_WITH_NO_FIXED_OPERANDS, TWO_OPERANDS_ONE_RESULT,
};
use super::attrs::{OPENING_BATCH_ATTRS, OPENING_CLAIM_ATTRS};

pub(super) fn validate_op(operation: OperationRef<'_, '_>, suffix: &str) -> MaybeValidation {
    let result = match suffix {
        "opening_claim" => attrs_shape(operation, OPENING_CLAIM_ATTRS, TWO_OPERANDS_ONE_RESULT),
        "opening_claim_equal" => opening_claim_equal(operation),
        "opening_batch" => attrs_counted_min_shape(
            operation,
            OPENING_BATCH_ATTRS,
            ORDERED_CLAIMS_WITH_NO_FIXED_OPERANDS,
        ),
        _ => return None,
    };
    Some(result)
}
