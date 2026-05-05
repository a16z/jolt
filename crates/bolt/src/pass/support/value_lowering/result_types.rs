pub(in crate::pass) const COMPUTE_OPENING_INPUT_RESULT_TYPES: &[&str] = &[
    "!compute.point",
    "!compute.field_value",
    "!compute.opening_claim_type",
];
pub(in crate::pass) const COMPUTE_POINT_RESULT_TYPES: &[&str] = &["!compute.point"];
pub(in crate::pass) const COMPUTE_FIELD_RESULT_TYPES: &[&str] = &["!compute.field_value"];

pub(in crate::pass) const CPU_OPENING_INPUT_RESULT_TYPES: &[&str] =
    &["!cpu.point", "!cpu.field_value", "!cpu.opening_claim_type"];
pub(in crate::pass) const CPU_POINT_RESULT_TYPES: &[&str] = &["!cpu.point"];
pub(in crate::pass) const CPU_FIELD_RESULT_TYPES: &[&str] = &["!cpu.field_value"];
