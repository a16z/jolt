pub(super) const FIELD_DEFINE_ATTRS: &[&str] = &["sym_name", "modulus_bits", "role"];
pub(super) const FIELD_CONST_ATTRS: &[&str] = &["sym_name", "field", "value"];
pub(super) const FIELD_UNIT_ATTRS: &[&str] = &["sym_name", "field"];
pub(super) const FIELD_BINARY_ATTRS: &[&str] = &["sym_name"];
pub(super) const FIELD_POW_ATTRS: &[&str] = &["sym_name", "exponent"];

pub(super) const HASH_FUNCTION_ATTRS: &[&str] = &["sym_name", "algorithm"];
pub(super) const TRANSCRIPT_SCHEME_ATTRS: &[&str] = &["sym_name", "hash"];
pub(super) const PCS_SCHEME_ATTRS: &[&str] = &["sym_name", "field"];

pub(super) const POLY_DOMAIN_ATTRS: &[&str] = &["sym_name", "field", "log_size"];
pub(super) const POINT_SLICE_ATTRS: &[&str] = &["sym_name", "source", "offset", "length"];
pub(super) const POINT_ZERO_ATTRS: &[&str] = &["sym_name", "field", "arity"];
pub(super) const POINT_CONCAT_ATTRS: &[&str] = &["sym_name", "layout", "arity"];
pub(super) const LAGRANGE_BASIS_EVAL_ATTRS: &[&str] =
    &["sym_name", "domain_start", "domain_size", "index"];

pub(super) const PROTOCOL_PARAMS_ATTRS: &[&str] = &["sym_name", "field", "pcs", "transcript"];
pub(super) const PROTOCOL_BOUNDARY_ATTRS: &[&str] = &["sym_name", "roles"];
pub(super) const PARTY_FUNCTION_ATTRS: &[&str] = &["sym_name", "source", "role"];
