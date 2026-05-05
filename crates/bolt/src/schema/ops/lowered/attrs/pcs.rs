pub(in crate::schema::ops::lowered) const PCS_COMMIT_BATCH_ATTRS: &[&str] = &[
    "sym_name",
    "artifact",
    "pcs",
    "oracle_family",
    "ordered_oracles",
    "label",
    "domain",
    "num_vars",
    "count",
];

pub(in crate::schema::ops::lowered) const PCS_COMMIT_OPTIONAL_ATTRS: &[&str] = &[
    "sym_name",
    "artifact",
    "pcs",
    "oracle",
    "label",
    "domain",
    "num_vars",
    "skip_policy",
];

pub(in crate::schema::ops::lowered) const PCS_OPENING_CLAIM_ATTRS: &[&str] =
    &["sym_name", "oracle", "family", "domain", "point_arity"];

pub(in crate::schema::ops::lowered) const PCS_OPENING_BATCH_ATTRS: &[&str] = &[
    "sym_name",
    "proof_slot",
    "policy",
    "count",
    "ordered_claims",
];

pub(in crate::schema::ops::lowered) const PCS_BATCH_OPENING_ATTRS: &[&str] =
    &["sym_name", "pcs", "proof_slot", "transcript_label"];
