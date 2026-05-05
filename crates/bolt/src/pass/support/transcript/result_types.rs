use melior::ir::operation::OperationRef;

use crate::mlir::MlirError;

use super::super::attrs::string_attr;
use super::super::diagnostic::schema_error;

pub(in crate::pass) const COMPUTE_TRANSCRIPT_STATE_RESULT_TYPES: &[&str] =
    &["!compute.transcript_state"];
pub(in crate::pass) const CPU_TRANSCRIPT_STATE_RESULT_TYPES: &[&str] = &["!cpu.transcript_state"];

pub(in crate::pass) fn transcript_squeeze_compute_result_types(
    operation: OperationRef<'_, '_>,
) -> Result<[&'static str; 2], MlirError> {
    Ok([
        "!compute.transcript_state",
        transcript_squeeze_value_type(
            string_attr(operation, "kind")?.as_str(),
            "!compute.point",
            "!compute.field_value",
        )?,
    ])
}

pub(in crate::pass) fn transcript_squeeze_cpu_result_types(
    operation: OperationRef<'_, '_>,
) -> Result<[&'static str; 2], MlirError> {
    Ok([
        "!cpu.transcript_state",
        transcript_squeeze_value_type(
            string_attr(operation, "kind")?.as_str(),
            "!cpu.point",
            "!cpu.field_value",
        )?,
    ])
}

fn transcript_squeeze_value_type(
    kind: &str,
    point_type: &'static str,
    scalar_type: &'static str,
) -> Result<&'static str, MlirError> {
    match kind {
        "challenge_vector" => Ok(point_type),
        "challenge_scalar" | "scalar" => Ok(scalar_type),
        kind => Err(schema_error(format!(
            "unsupported transcript squeeze kind `{kind}`"
        ))),
    }
}
