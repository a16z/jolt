pub(in crate::pass) const SUMCHECK_CLAIM_ATTRS: &[&str] = &[
    "stage",
    "domain",
    "num_rounds",
    "degree",
    "claim",
    "relation",
];
pub(in crate::pass) const SUMCHECK_BATCH_ATTRS: &[&str] = &[
    "stage",
    "proof_slot",
    "policy",
    "count",
    "ordered_claims",
    "claim_label",
    "round_label",
    "round_schedule",
];
pub(in crate::pass) const SUMCHECK_DRIVER_ATTRS: &[&str] = &[
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

pub(in crate::pass) const SUMCHECK_KERNEL_CLAIM_SOURCE_ATTRS: &[&str] =
    &["stage", "domain", "num_rounds", "degree", "claim"];
pub(in crate::pass) const SUMCHECK_KERNEL_DRIVER_SOURCE_ATTRS: &[&str] = &[
    "stage",
    "proof_slot",
    "policy",
    "round_schedule",
    "claim_label",
    "round_label",
    "num_rounds",
    "degree",
];

pub(in crate::pass) const SUMCHECK_KERNEL_CLAIM_ATTRS: &[&str] =
    &["stage", "domain", "num_rounds", "degree", "claim", "kernel"];
pub(in crate::pass) const SUMCHECK_KERNEL_DRIVER_ATTRS: &[&str] = &[
    "stage",
    "proof_slot",
    "kernel",
    "policy",
    "round_schedule",
    "claim_label",
    "round_label",
    "num_rounds",
    "degree",
];
