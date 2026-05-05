pub(in crate::schema::ops::lowered) const OPENING_CLAIM_ATTRS: &[&str] =
    &["sym_name", "oracle", "domain", "point_arity", "claim_kind"];

pub(in crate::schema::ops::lowered) const OPENING_BATCH_ATTRS: &[&str] = &[
    "sym_name",
    "stage",
    "proof_slot",
    "policy",
    "count",
    "ordered_claims",
];
