pub(in crate::schema::ops::lowered) const OPENING_INPUT_ATTRS: &[&str] = &[
    "sym_name",
    "source_stage",
    "source_claim",
    "oracle",
    "domain",
    "point_arity",
    "claim_kind",
];

pub(in crate::schema::ops::lowered) const POINT_SLICE_ATTRS: &[&str] =
    &["sym_name", "source", "offset", "length"];
pub(in crate::schema::ops::lowered) const POINT_ZERO_ATTRS: &[&str] =
    &["sym_name", "field", "arity"];
pub(in crate::schema::ops::lowered) const POINT_CONCAT_ATTRS: &[&str] =
    &["sym_name", "layout", "arity"];

pub(in crate::schema::ops::lowered) const FIELD_CONST_ATTRS: &[&str] =
    &["sym_name", "field", "value"];
pub(in crate::schema::ops::lowered) const FIELD_UNIT_ATTRS: &[&str] = &["sym_name", "field"];
pub(in crate::schema::ops::lowered) const FIELD_BINARY_ATTRS: &[&str] = &["sym_name"];
pub(in crate::schema::ops::lowered) const FIELD_POW_ATTRS: &[&str] = &["sym_name", "exponent"];
pub(in crate::schema::ops::lowered) const LAGRANGE_BASIS_EVAL_ATTRS: &[&str] =
    &["sym_name", "domain_start", "domain_size", "index"];
