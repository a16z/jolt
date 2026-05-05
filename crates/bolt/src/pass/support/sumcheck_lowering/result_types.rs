pub(in crate::pass) const COMPUTE_SUMCHECK_EVAL_RESULT_TYPES: &[&str] = &["!compute.field_value"];
pub(in crate::pass) const COMPUTE_SUMCHECK_INSTANCE_RESULT_TYPES: &[&str] =
    &["!compute.point", "!compute.sumcheck_result_type"];

pub(in crate::pass) const CPU_SUMCHECK_EVAL_RESULT_TYPES: &[&str] = &["!cpu.field_value"];
pub(in crate::pass) const CPU_SUMCHECK_INSTANCE_RESULT_TYPES: &[&str] =
    &["!cpu.point", "!cpu.sumcheck_result_type"];
