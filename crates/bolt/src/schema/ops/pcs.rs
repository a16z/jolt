mod notation;

use melior::ir::OperationRef;

use super::support::{
    attrs_counted_min_shape, attrs_shape, MaybeValidation, ORDERED_CLAIMS_WITH_NO_FIXED_OPERANDS,
    TWO_OPERANDS_ONE_RESULT, TWO_OPERANDS_TWO_RESULTS,
};
use notation::{PCS_BATCH_OPENING_ATTRS, PCS_OPENING_BATCH_ATTRS, PCS_OPENING_CLAIM_ATTRS};

pub(super) fn validate_op(operation: OperationRef<'_, '_>, name: &str) -> MaybeValidation {
    let result = match name {
        "pcs.opening_claim" => {
            attrs_shape(operation, PCS_OPENING_CLAIM_ATTRS, TWO_OPERANDS_ONE_RESULT)
        }
        "pcs.opening_batch" => attrs_counted_min_shape(
            operation,
            PCS_OPENING_BATCH_ATTRS,
            ORDERED_CLAIMS_WITH_NO_FIXED_OPERANDS,
        ),
        "pcs.batch_open" | "pcs.batch_verify" => {
            attrs_shape(operation, PCS_BATCH_OPENING_ATTRS, TWO_OPERANDS_TWO_RESULTS)
        }
        _ => return None,
    };
    Some(result)
}
