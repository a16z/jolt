use melior::ir::operation::OperationRef;

use crate::ir::Compute;
use crate::schema::operation_name;

use super::super::super::support::{
    ValueDialect, ValueOpFamily, COMPUTE_FIELD_RESULT_TYPES, COMPUTE_OPENING_INPUT_RESULT_TYPES,
    COMPUTE_POINT_RESULT_TYPES,
};

pub(super) struct PartyToComputeValueDialect;

impl ValueDialect for PartyToComputeValueDialect {
    type Phase = Compute;

    const OPENING_INPUT_RESULT_TYPES: &'static [&'static str] = COMPUTE_OPENING_INPUT_RESULT_TYPES;
    const POINT_RESULT_TYPES: &'static [&'static str] = COMPUTE_POINT_RESULT_TYPES;
    const FIELD_RESULT_TYPES: &'static [&'static str] = COMPUTE_FIELD_RESULT_TYPES;

    fn classify(source_name: &str) -> Option<ValueOpFamily> {
        match source_name {
            "piop.opening_input" => Some(ValueOpFamily::OpeningInput),
            "poly.point_slice" => Some(ValueOpFamily::PointSlice),
            "poly.point_zero" => Some(ValueOpFamily::PointZero),
            "poly.point_concat" => Some(ValueOpFamily::PointConcat),
            "field.const" => Some(ValueOpFamily::FieldConst),
            "field.zero" | "field.one" => Some(ValueOpFamily::FieldUnit),
            "field.add"
            | "field.sub"
            | "field.mul"
            | "field.neg"
            | "field.pow"
            | "poly.lagrange_basis_eval" => Some(ValueOpFamily::FieldExpression),
            _ => None,
        }
    }

    fn target_op_name(operation: OperationRef<'_, '_>) -> String {
        match operation_name(operation).as_str() {
            "piop.opening_input" => "compute.opening_input".to_owned(),
            "poly.point_slice" => "compute.point_slice".to_owned(),
            "poly.point_zero" => "compute.point_zero".to_owned(),
            "poly.point_concat" => "compute.point_concat".to_owned(),
            "poly.lagrange_basis_eval" => "compute.poly_lagrange_basis_eval".to_owned(),
            source_name => format!("compute.{}", source_name.replace('.', "_")),
        }
    }
}
