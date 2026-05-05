use melior::ir::operation::OperationRef;

use crate::mlir::MlirError;
use crate::schema::operation_name;

use super::super::attrs::copy_attrs;

pub(super) const OPENING_INPUT_ATTRS: &[&str] = &[
    "source_stage",
    "source_claim",
    "oracle",
    "domain",
    "point_arity",
    "claim_kind",
];
pub(super) const POINT_SLICE_ATTRS: &[&str] = &["source", "offset", "length"];
pub(super) const POINT_ZERO_ATTRS: &[&str] = &["field", "arity"];
pub(super) const POINT_CONCAT_ATTRS: &[&str] = &["layout", "arity"];
pub(super) const FIELD_CONST_ATTRS: &[&str] = &["field", "value"];
pub(super) const FIELD_UNIT_ATTRS: &[&str] = &["field"];
const FIELD_POW_ATTRS: &[&str] = &["exponent"];
const LAGRANGE_BASIS_EVAL_ATTRS: &[&str] = &["domain_start", "domain_size", "index"];

pub(super) fn field_expression_attrs(
    operation: OperationRef<'_, '_>,
) -> Result<Vec<(String, String)>, MlirError> {
    match operation_name(operation).as_str() {
        "field.pow" | "compute.field_pow" => copy_attrs(operation, FIELD_POW_ATTRS),
        "poly.lagrange_basis_eval" | "compute.poly_lagrange_basis_eval" => {
            copy_attrs(operation, LAGRANGE_BASIS_EVAL_ATTRS)
        }
        _ => Ok(Vec::new()),
    }
}
