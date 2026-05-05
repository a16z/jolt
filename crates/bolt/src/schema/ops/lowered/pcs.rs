use melior::ir::OperationRef;

use super::super::support::{
    attrs_counted_min_shape, attrs_shape, MaybeValidation, ONE_OPERAND_ONE_RESULT,
    ORDERED_CLAIMS_WITH_NO_FIXED_OPERANDS, TWO_OPERANDS_ONE_RESULT, TWO_OPERANDS_TWO_RESULTS,
};
use super::attrs::{
    PCS_BATCH_OPENING_ATTRS, PCS_COMMIT_BATCH_ATTRS, PCS_COMMIT_OPTIONAL_ATTRS,
    PCS_OPENING_BATCH_ATTRS, PCS_OPENING_CLAIM_ATTRS,
};

pub(super) fn validate_op(operation: OperationRef<'_, '_>, suffix: &str) -> MaybeValidation {
    let result = match suffix {
        "pcs_commit_batch" | "pcs_receive_batch" => {
            attrs_shape(operation, PCS_COMMIT_BATCH_ATTRS, ONE_OPERAND_ONE_RESULT)
        }
        "pcs_commit_optional" | "pcs_receive_optional" => {
            attrs_shape(operation, PCS_COMMIT_OPTIONAL_ATTRS, ONE_OPERAND_ONE_RESULT)
        }
        "pcs_opening_claim" => {
            attrs_shape(operation, PCS_OPENING_CLAIM_ATTRS, TWO_OPERANDS_ONE_RESULT)
        }
        "pcs_opening_batch" => attrs_counted_min_shape(
            operation,
            PCS_OPENING_BATCH_ATTRS,
            ORDERED_CLAIMS_WITH_NO_FIXED_OPERANDS,
        ),
        "pcs_batch_open" | "pcs_batch_verify" => {
            attrs_shape(operation, PCS_BATCH_OPENING_ATTRS, TWO_OPERANDS_TWO_RESULTS)
        }
        _ => return None,
    };
    Some(result)
}
