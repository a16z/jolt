use melior::ir::OperationRef;

use super::super::constraints::{
    require_attrs, require_counted_operands, require_min_shape, require_opening_claim_equality,
    require_shape,
};
use super::super::SchemaError;

mod shape;

pub(super) use shape::{
    CountedOperandShape, ExactOpShape, MinOpShape, AT_LEAST_ONE_OPERAND_ONE_RESULT,
    NO_OPERANDS_ONE_RESULT, NO_OPERANDS_THREE_RESULTS, ONE_OPERAND_NO_RESULTS,
    ONE_OPERAND_ONE_RESULT, ONE_OPERAND_TWO_RESULTS, ORDERED_CLAIMS_WITH_NO_FIXED_OPERANDS,
    ORDERED_CLAIMS_WITH_ONE_FIXED_OPERAND, TWO_OPERANDS_FOUR_RESULTS, TWO_OPERANDS_NO_RESULTS,
    TWO_OPERANDS_ONE_RESULT, TWO_OPERANDS_TWO_RESULTS,
};

pub(super) type Validation = Result<(), SchemaError>;
pub(super) type MaybeValidation = Option<Validation>;

const OPENING_CLAIM_EQUAL_ATTRS: &[&str] = &["sym_name", "mode"];

pub(super) fn attrs(operation: OperationRef<'_, '_>, attrs: &[&str]) -> Validation {
    require_attrs(operation, attrs)
}

pub(super) fn shape(operation: OperationRef<'_, '_>, shape: ExactOpShape) -> Validation {
    require_shape(operation, shape.operands, shape.results)
}

pub(super) fn min_shape(operation: OperationRef<'_, '_>, shape: MinOpShape) -> Validation {
    require_min_shape(operation, shape.min_operands, shape.results)
}

pub(super) fn attrs_shape(
    operation: OperationRef<'_, '_>,
    attrs: &[&str],
    shape: ExactOpShape,
) -> Validation {
    require_attrs(operation, attrs)?;
    self::shape(operation, shape)
}

pub(super) fn attrs_min_shape(
    operation: OperationRef<'_, '_>,
    attrs: &[&str],
    shape: MinOpShape,
) -> Validation {
    require_attrs(operation, attrs)?;
    min_shape(operation, shape)
}

pub(super) fn attrs_counted_min_shape(
    operation: OperationRef<'_, '_>,
    attrs: &[&str],
    shape: CountedOperandShape,
) -> Validation {
    require_attrs(operation, attrs)?;
    require_min_shape(operation, shape.min_operands, shape.results)?;
    require_counted_operands(operation, shape.fixed_operands, shape.ordered_attr)
}

pub(super) fn opening_claim_equal(operation: OperationRef<'_, '_>) -> Validation {
    require_attrs(operation, OPENING_CLAIM_EQUAL_ATTRS)?;
    shape(operation, TWO_OPERANDS_NO_RESULTS)?;
    require_opening_claim_equality(operation)
}
