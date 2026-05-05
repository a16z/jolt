use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Cpu};
use crate::mlir::{MeliorContext, MlirError};

use super::super::super::support::{
    append_lowered_result_count, compute_to_cpu_op_name, LoweredResultCount,
};

const ATTRS: &[&str] = &[
    "oracle",
    "source",
    "domain",
    "num_vars",
    "trace_num_vars",
    "chunk",
    "num_chunks",
    "chunk_bits",
    "padding",
    "layout",
    "skip_policy",
];
const RESULT_TYPES: &[&str] = &["!cpu.oracle_buffer"];

pub(super) fn lower<'c, 'a>(
    context: &'c MeliorContext,
    cpu: &'a BoltModule<'c, Cpu>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
) -> Result<(), MlirError> {
    let target_op = compute_to_cpu_op_name(op);
    append_lowered_result_count(
        context,
        cpu,
        value_map,
        op,
        0,
        &target_op,
        ATTRS,
        RESULT_TYPES,
        LoweredResultCount::One,
    )
}
