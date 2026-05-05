use melior::ir::OperationRef;

mod attrs;

use super::super::super::constraints::require_attrs;
use super::super::support::{
    attrs_counted_min_shape, attrs_shape, min_shape, shape, MaybeValidation, Validation,
    AT_LEAST_ONE_OPERAND_ONE_RESULT, ONE_OPERAND_ONE_RESULT, ORDERED_CLAIMS_WITH_NO_FIXED_OPERANDS,
    TWO_OPERANDS_FOUR_RESULTS, TWO_OPERANDS_TWO_RESULTS,
};
use super::dialect::LoweredDialect;
use attrs::{
    BATCH_ATTRS, CLAIM_BEFORE_REF_ATTRS, DRIVER_AFTER_REF_ATTRS, DRIVER_BEFORE_REF_ATTRS,
    EVAL_ATTRS, INSTANCE_RESULT_ATTRS,
};

pub(super) fn validate_op<D: LoweredDialect>(
    operation: OperationRef<'_, '_>,
    suffix: &str,
) -> MaybeValidation {
    let result = match suffix {
        "sumcheck_claim" => claim(operation, D::PRIMARY_SUMCHECK_REFERENCE_ATTR),
        "sumcheck_kernel_claim" if D::CAPABILITIES.has_kernel_sumcheck_ops() => {
            claim(operation, "kernel")
        }
        "sumcheck_verify_claim" => claim(operation, "relation"),
        "sumcheck_batch" => batch(operation),
        "sumcheck_driver" => driver(operation, D::PRIMARY_SUMCHECK_REFERENCE_ATTR),
        "sumcheck_kernel_driver" if D::CAPABILITIES.has_kernel_sumcheck_ops() => {
            driver(operation, "kernel")
        }
        "sumcheck_verify" => driver(operation, "relation"),
        "sumcheck_eval" => eval(operation),
        "sumcheck_instance_result" => instance_result(operation),
        _ => return None,
    };
    Some(result)
}

fn claim(operation: OperationRef<'_, '_>, reference_attr: &str) -> Validation {
    require_attrs(operation, CLAIM_BEFORE_REF_ATTRS)?;
    require_attrs(operation, &[reference_attr])?;
    min_shape(operation, AT_LEAST_ONE_OPERAND_ONE_RESULT)
}

fn batch(operation: OperationRef<'_, '_>) -> Validation {
    attrs_counted_min_shape(
        operation,
        BATCH_ATTRS,
        ORDERED_CLAIMS_WITH_NO_FIXED_OPERANDS,
    )
}

fn driver(operation: OperationRef<'_, '_>, reference_attr: &str) -> Validation {
    require_attrs(operation, DRIVER_BEFORE_REF_ATTRS)?;
    require_attrs(operation, &[reference_attr])?;
    require_attrs(operation, DRIVER_AFTER_REF_ATTRS)?;
    shape(operation, TWO_OPERANDS_FOUR_RESULTS)
}

fn eval(operation: OperationRef<'_, '_>) -> Validation {
    attrs_shape(operation, EVAL_ATTRS, ONE_OPERAND_ONE_RESULT)
}

fn instance_result(operation: OperationRef<'_, '_>) -> Validation {
    attrs_shape(operation, INSTANCE_RESULT_ATTRS, TWO_OPERANDS_TWO_RESULTS)
}
