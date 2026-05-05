mod pcs;

use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Cpu};
use crate::mlir::{MeliorContext, MlirError};
use crate::schema::operation_name;

pub(super) const COMMITMENT_ARTIFACT_RESULT_TYPES: &[&str] = &["!cpu.commitment_artifact"];

pub(super) fn lower_op<'c, 'a>(
    context: &'c MeliorContext,
    cpu: &'a BoltModule<'c, Cpu>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
) -> Result<bool, MlirError> {
    match operation_name(op).as_str() {
        "compute.pcs_commit_batch" | "compute.pcs_receive_batch" => {
            pcs::lower_batch(context, cpu, value_map, op)?;
        }
        "compute.pcs_commit_optional" | "compute.pcs_receive_optional" => {
            pcs::lower_optional(context, cpu, value_map, op)?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}
