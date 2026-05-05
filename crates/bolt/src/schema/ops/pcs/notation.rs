pub(super) const PCS_OPENING_CLAIM_ATTRS: &[&str] =
    &["sym_name", "oracle", "family", "domain", "point_arity"];

pub(super) const PCS_OPENING_BATCH_ATTRS: &[&str] = &[
    "sym_name",
    "proof_slot",
    "policy",
    "count",
    "ordered_claims",
];

pub(super) const PCS_BATCH_OPENING_ATTRS: &[&str] =
    &["sym_name", "pcs", "proof_slot", "transcript_label"];
