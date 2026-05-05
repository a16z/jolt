pub(in crate::pass) const COMPUTE_SUMCHECK_CLAIM_RESULT_TYPES: &[&str] =
    &["!compute.sumcheck_claim_type"];
pub(in crate::pass) const COMPUTE_SUMCHECK_BATCH_RESULT_TYPES: &[&str] =
    &["!compute.sumcheck_batch_type"];
pub(in crate::pass) const COMPUTE_SUMCHECK_DRIVER_RESULT_TYPES: &[&str] = &[
    "!compute.transcript_state",
    "!compute.point",
    "!compute.sumcheck_result_type",
    "!compute.sumcheck_proof_type",
];

pub(in crate::pass) const CPU_SUMCHECK_CLAIM_RESULT_TYPES: &[&str] = &["!cpu.sumcheck_claim_type"];
pub(in crate::pass) const CPU_SUMCHECK_BATCH_RESULT_TYPES: &[&str] = &["!cpu.sumcheck_batch_type"];
pub(in crate::pass) const CPU_SUMCHECK_DRIVER_RESULT_TYPES: &[&str] = &[
    "!cpu.transcript_state",
    "!cpu.point",
    "!cpu.sumcheck_result_type",
    "!cpu.sumcheck_proof_type",
];
