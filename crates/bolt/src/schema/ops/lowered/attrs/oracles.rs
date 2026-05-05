pub(in crate::schema::ops::lowered) const DENSE_TRACE_ATTRS: &[&str] = &[
    "sym_name", "oracle", "source", "domain", "num_vars", "padding",
];

pub(in crate::schema::ops::lowered) const ONE_HOT_CHUNK_ATTRS: &[&str] = &[
    "sym_name",
    "oracle",
    "source",
    "domain",
    "num_vars",
    "trace_num_vars",
    "chunk",
    "num_chunks",
    "chunk_bits",
    "padding",
    "layout",
];

pub(in crate::schema::ops::lowered) const OPTIONAL_ADVICE_ATTRS: &[&str] = &[
    "sym_name",
    "oracle",
    "source",
    "domain",
    "num_vars",
    "skip_policy",
];

pub(in crate::schema::ops::lowered) const ORACLE_REF_ATTRS: &[&str] =
    &["sym_name", "oracle", "domain", "num_vars"];
pub(in crate::schema::ops::lowered) const ORACLE_FAMILY_INIT_ATTRS: &[&str] =
    &["sym_name", "family", "count"];
pub(in crate::schema::ops::lowered) const ORACLE_FAMILY_APPEND_ATTRS: &[&str] =
    &["sym_name", "family", "oracle", "index"];
