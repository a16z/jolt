pub(in crate::pass) const COMPUTE_OPENING_CLAIM_RESULT_TYPES: &[&str] =
    &["!compute.opening_claim_type"];
pub(in crate::pass) const COMPUTE_OPENING_BATCH_RESULT_TYPES: &[&str] =
    &["!compute.opening_batch_type"];
pub(in crate::pass) const COMPUTE_OPENING_BATCH_OPENING_RESULT_TYPES: &[&str] =
    &["!compute.transcript_state", "!compute.opening_proof_type"];

pub(in crate::pass) const CPU_OPENING_CLAIM_RESULT_TYPES: &[&str] = &["!cpu.opening_claim_type"];
pub(in crate::pass) const CPU_OPENING_BATCH_RESULT_TYPES: &[&str] = &["!cpu.opening_batch_type"];
pub(in crate::pass) const CPU_OPENING_BATCH_OPENING_RESULT_TYPES: &[&str] =
    &["!cpu.transcript_state", "!cpu.opening_proof_type"];
