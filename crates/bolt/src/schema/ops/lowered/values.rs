use melior::ir::OperationRef;

use super::super::support::{
    attrs_min_shape, attrs_shape, MaybeValidation, AT_LEAST_ONE_OPERAND_ONE_RESULT,
    NO_OPERANDS_ONE_RESULT, NO_OPERANDS_THREE_RESULTS, ONE_OPERAND_ONE_RESULT,
    TWO_OPERANDS_ONE_RESULT,
};
use super::attrs::{
    FIELD_BINARY_ATTRS, FIELD_CONST_ATTRS, FIELD_POW_ATTRS, FIELD_UNIT_ATTRS,
    LAGRANGE_BASIS_EVAL_ATTRS, OPENING_INPUT_ATTRS, POINT_CONCAT_ATTRS, POINT_SLICE_ATTRS,
    POINT_ZERO_ATTRS,
};

pub(super) fn validate_op(operation: OperationRef<'_, '_>, suffix: &str) -> MaybeValidation {
    let result = match suffix {
        "opening_input" => attrs_shape(operation, OPENING_INPUT_ATTRS, NO_OPERANDS_THREE_RESULTS),
        "point_slice" => attrs_shape(operation, POINT_SLICE_ATTRS, ONE_OPERAND_ONE_RESULT),
        "point_zero" => attrs_shape(operation, POINT_ZERO_ATTRS, NO_OPERANDS_ONE_RESULT),
        "point_concat" => attrs_min_shape(
            operation,
            POINT_CONCAT_ATTRS,
            AT_LEAST_ONE_OPERAND_ONE_RESULT,
        ),
        "field_const" => attrs_shape(operation, FIELD_CONST_ATTRS, NO_OPERANDS_ONE_RESULT),
        "field_zero" | "field_one" => {
            attrs_shape(operation, FIELD_UNIT_ATTRS, NO_OPERANDS_ONE_RESULT)
        }
        "field_add" | "field_sub" | "field_mul" => {
            attrs_shape(operation, FIELD_BINARY_ATTRS, TWO_OPERANDS_ONE_RESULT)
        }
        "field_neg" => attrs_shape(operation, FIELD_BINARY_ATTRS, ONE_OPERAND_ONE_RESULT),
        "field_pow" => attrs_shape(operation, FIELD_POW_ATTRS, ONE_OPERAND_ONE_RESULT),
        "poly_lagrange_basis_eval" => {
            attrs_shape(operation, LAGRANGE_BASIS_EVAL_ATTRS, ONE_OPERAND_ONE_RESULT)
        }
        _ => return None,
    };
    Some(result)
}
