pub(super) const CLAIM_BEFORE_REF_ATTRS: &[&str] = &[
    "sym_name",
    "stage",
    "domain",
    "num_rounds",
    "degree",
    "claim",
];

pub(super) const BATCH_ATTRS: &[&str] = &[
    "sym_name",
    "stage",
    "proof_slot",
    "policy",
    "count",
    "ordered_claims",
    "claim_label",
    "round_label",
    "round_schedule",
];

pub(super) const DRIVER_BEFORE_REF_ATTRS: &[&str] = &["sym_name", "stage", "proof_slot"];

pub(super) const DRIVER_AFTER_REF_ATTRS: &[&str] = &[
    "policy",
    "round_schedule",
    "claim_label",
    "round_label",
    "num_rounds",
    "degree",
];

pub(super) const EVAL_ATTRS: &[&str] = &["sym_name", "source", "name", "index", "oracle"];

pub(super) const INSTANCE_RESULT_ATTRS: &[&str] = &[
    "sym_name",
    "source",
    "claim",
    "relation",
    "index",
    "point_arity",
    "num_rounds",
    "round_offset",
    "point_order",
    "degree",
];
