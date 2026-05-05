use super::family::ValueOpFamily;

pub(in crate::pass) fn classify_compute_value_op(source_name: &str) -> Option<ValueOpFamily> {
    match source_name {
        "compute.opening_input" => Some(ValueOpFamily::OpeningInput),
        "compute.point_slice" => Some(ValueOpFamily::PointSlice),
        "compute.point_zero" => Some(ValueOpFamily::PointZero),
        "compute.point_concat" => Some(ValueOpFamily::PointConcat),
        "compute.field_const" => Some(ValueOpFamily::FieldConst),
        "compute.field_zero" | "compute.field_one" => Some(ValueOpFamily::FieldUnit),
        "compute.field_add"
        | "compute.field_sub"
        | "compute.field_mul"
        | "compute.field_neg"
        | "compute.field_pow"
        | "compute.poly_lagrange_basis_eval" => Some(ValueOpFamily::FieldExpression),
        _ => None,
    }
}
