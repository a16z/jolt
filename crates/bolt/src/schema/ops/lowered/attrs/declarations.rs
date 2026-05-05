pub(in crate::schema::ops::lowered) const RELATION_ATTRS: &[&str] = &[
    "sym_name",
    "kind",
    "domain",
    "num_rounds",
    "degree",
    "output_count",
];

pub(in crate::schema::ops::lowered) const KERNEL_ATTRS: &[&str] =
    &["sym_name", "relation", "kind", "backend", "abi"];

pub(in crate::schema::ops::lowered) const PARAMS_ATTRS: &[&str] =
    &["sym_name", "field", "pcs", "transcript"];
pub(in crate::schema::ops::lowered) const FUNCTION_ATTRS: &[&str] = &["sym_name", "source"];
