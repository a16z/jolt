pub(super) const RELATION_ATTRS: &[&str] = &[
    "sym_name",
    "kind",
    "domain",
    "num_rounds",
    "degree",
    "output_count",
];

pub(super) const ORACLE_ATTRS: &[&str] = &[
    "sym_name",
    "field",
    "domain",
    "commit_domain",
    "visibility",
    "layout",
];

pub(super) const ORACLE_FAMILY_ATTRS: &[&str] = &[
    "sym_name",
    "ordered_oracles",
    "visibility",
    "count",
    "domain",
];

pub(super) const COMMIT_PUBLISH_BATCH_ATTRS: &[&str] = &["sym_name", "oracle_family", "label"];
pub(super) const COMMIT_PUBLISH_OPTIONAL_ATTRS: &[&str] =
    &["sym_name", "oracle", "label", "skip_policy"];
pub(super) const PCS_COMMIT_BATCH_ATTRS: &[&str] = &["sym_name", "scheme"];

pub(super) const TRANSCRIPT_ABSORB_ATTRS: &[&str] = &["sym_name", "label"];
pub(super) const TRANSCRIPT_ABSORB_BYTES_ATTRS: &[&str] = &["sym_name", "label", "payload"];
pub(super) const TRANSCRIPT_SQUEEZE_ATTRS: &[&str] = &["sym_name", "label", "kind", "count"];
pub(super) const TRANSCRIPT_STATE_ATTRS: &[&str] = &["sym_name", "scheme"];

pub(super) const SUMCHECK_CLAIM_ATTRS: &[&str] = &[
    "sym_name",
    "stage",
    "domain",
    "num_rounds",
    "degree",
    "claim",
    "relation",
];

pub(super) const OPENING_INPUT_ATTRS: &[&str] = &[
    "sym_name",
    "source_stage",
    "source_claim",
    "oracle",
    "domain",
    "point_arity",
    "claim_kind",
];

pub(super) const SUMCHECK_BATCH_ATTRS: &[&str] = &[
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

pub(super) const SUMCHECK_DRIVER_ATTRS: &[&str] = &[
    "sym_name",
    "stage",
    "proof_slot",
    "relation",
    "policy",
    "round_schedule",
    "claim_label",
    "round_label",
    "num_rounds",
    "degree",
];

pub(super) const SUMCHECK_EVAL_ATTRS: &[&str] = &["sym_name", "source", "name", "index", "oracle"];

pub(super) const SUMCHECK_INSTANCE_RESULT_ATTRS: &[&str] = &[
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

pub(super) const OPENING_CLAIM_ATTRS: &[&str] =
    &["sym_name", "oracle", "domain", "point_arity", "claim_kind"];

pub(super) const OPENING_BATCH_ATTRS: &[&str] = &[
    "sym_name",
    "stage",
    "proof_slot",
    "policy",
    "count",
    "ordered_claims",
];
