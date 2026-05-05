use melior::ir::OperationRef;

use super::super::support::{
    attrs_counted_min_shape, attrs_min_shape, attrs_shape, MaybeValidation,
    AT_LEAST_ONE_OPERAND_ONE_RESULT, ONE_OPERAND_ONE_RESULT, ORDERED_CLAIMS_WITH_ONE_FIXED_OPERAND,
    TWO_OPERANDS_FOUR_RESULTS, TWO_OPERANDS_TWO_RESULTS,
};
use super::notation::{
    SUMCHECK_BATCH_ATTRS, SUMCHECK_CLAIM_ATTRS, SUMCHECK_DRIVER_ATTRS, SUMCHECK_EVAL_ATTRS,
    SUMCHECK_INSTANCE_RESULT_ATTRS,
};

pub(super) fn validate_op(operation: OperationRef<'_, '_>, name: &str) -> MaybeValidation {
    let result = match name {
        "piop.sumcheck_claim" => attrs_min_shape(
            operation,
            SUMCHECK_CLAIM_ATTRS,
            AT_LEAST_ONE_OPERAND_ONE_RESULT,
        ),
        "piop.sumcheck_batch" => attrs_counted_min_shape(
            operation,
            SUMCHECK_BATCH_ATTRS,
            ORDERED_CLAIMS_WITH_ONE_FIXED_OPERAND,
        ),
        "piop.sumcheck" => attrs_shape(operation, SUMCHECK_DRIVER_ATTRS, TWO_OPERANDS_FOUR_RESULTS),
        "piop.sumcheck_eval" => attrs_shape(operation, SUMCHECK_EVAL_ATTRS, ONE_OPERAND_ONE_RESULT),
        "piop.sumcheck_instance_result" => attrs_shape(
            operation,
            SUMCHECK_INSTANCE_RESULT_ATTRS,
            TWO_OPERANDS_TWO_RESULTS,
        ),
        _ => return None,
    };
    Some(result)
}
