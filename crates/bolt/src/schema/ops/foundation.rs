mod notation;

use melior::ir::OperationRef;

use super::support::{attrs, attrs_min_shape, attrs_shape, MaybeValidation};
use super::support::{
    AT_LEAST_ONE_OPERAND_ONE_RESULT, NO_OPERANDS_ONE_RESULT, ONE_OPERAND_ONE_RESULT,
    TWO_OPERANDS_ONE_RESULT,
};
use notation::{
    FIELD_BINARY_ATTRS, FIELD_CONST_ATTRS, FIELD_DEFINE_ATTRS, FIELD_POW_ATTRS, FIELD_UNIT_ATTRS,
    HASH_FUNCTION_ATTRS, LAGRANGE_BASIS_EVAL_ATTRS, PARTY_FUNCTION_ATTRS, PCS_SCHEME_ATTRS,
    POINT_CONCAT_ATTRS, POINT_SLICE_ATTRS, POINT_ZERO_ATTRS, POLY_DOMAIN_ATTRS,
    PROTOCOL_BOUNDARY_ATTRS, PROTOCOL_PARAMS_ATTRS, TRANSCRIPT_SCHEME_ATTRS,
};

pub(super) fn validate_op(operation: OperationRef<'_, '_>, name: &str) -> MaybeValidation {
    let result = match name {
        "field.define" => attrs(operation, FIELD_DEFINE_ATTRS),
        "field.const" => attrs_shape(operation, FIELD_CONST_ATTRS, NO_OPERANDS_ONE_RESULT),
        "field.zero" | "field.one" => {
            attrs_shape(operation, FIELD_UNIT_ATTRS, NO_OPERANDS_ONE_RESULT)
        }
        "field.add" | "field.sub" | "field.mul" => {
            attrs_shape(operation, FIELD_BINARY_ATTRS, TWO_OPERANDS_ONE_RESULT)
        }
        "field.neg" => attrs_shape(operation, FIELD_BINARY_ATTRS, ONE_OPERAND_ONE_RESULT),
        "field.pow" => attrs_shape(operation, FIELD_POW_ATTRS, ONE_OPERAND_ONE_RESULT),
        "hash.function" => attrs(operation, HASH_FUNCTION_ATTRS),
        "transcript.scheme" => attrs(operation, TRANSCRIPT_SCHEME_ATTRS),
        "pcs.scheme" => attrs(operation, PCS_SCHEME_ATTRS),
        "poly.domain" => attrs(operation, POLY_DOMAIN_ATTRS),
        "poly.point_slice" => attrs_shape(operation, POINT_SLICE_ATTRS, ONE_OPERAND_ONE_RESULT),
        "poly.point_zero" => attrs_shape(operation, POINT_ZERO_ATTRS, NO_OPERANDS_ONE_RESULT),
        "poly.point_concat" => attrs_min_shape(
            operation,
            POINT_CONCAT_ATTRS,
            AT_LEAST_ONE_OPERAND_ONE_RESULT,
        ),
        "poly.lagrange_basis_eval" => {
            attrs_shape(operation, LAGRANGE_BASIS_EVAL_ATTRS, ONE_OPERAND_ONE_RESULT)
        }
        "protocol.params" => attrs(operation, PROTOCOL_PARAMS_ATTRS),
        "protocol.boundary" => attrs(operation, PROTOCOL_BOUNDARY_ATTRS),
        "party.function" => attrs(operation, PARTY_FUNCTION_ATTRS),
        _ => return None,
    };
    Some(result)
}
