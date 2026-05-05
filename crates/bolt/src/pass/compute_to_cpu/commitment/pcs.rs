mod notation;

use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Cpu};
use crate::mlir::MeliorContext;
use crate::mlir::MlirError;

use super::super::super::support::{
    append_and_map_result_count, required_lowered_operand, string_attr, LoweredResultCount,
};
use super::COMMITMENT_ARTIFACT_RESULT_TYPES;
use notation::{batch_attrs, batch_op_name, optional_attrs, optional_op_name};

pub(super) fn lower_batch<'c, 'a>(
    context: &'c MeliorContext,
    cpu: &'a BoltModule<'c, Cpu>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
) -> Result<(), MlirError> {
    let target_op = batch_op_name(op);
    let attrs = batch_attrs(op)?;
    let oracles = required_lowered_operand(
        op,
        value_map,
        0,
        "compute.pcs batch oracle family was not lowered",
    )?;
    let symbol = string_attr(op, "sym_name")?;
    append_and_map_result_count(
        context,
        cpu,
        value_map,
        op,
        target_op,
        &symbol,
        &attrs,
        &[oracles],
        COMMITMENT_ARTIFACT_RESULT_TYPES,
        LoweredResultCount::One,
    )
}

pub(super) fn lower_optional<'c, 'a>(
    context: &'c MeliorContext,
    cpu: &'a BoltModule<'c, Cpu>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
) -> Result<(), MlirError> {
    let target_op = optional_op_name(op);
    let attrs = optional_attrs(op)?;
    let oracle = required_lowered_operand(
        op,
        value_map,
        0,
        "compute.pcs optional oracle was not lowered",
    )?;
    let symbol = string_attr(op, "sym_name")?;
    append_and_map_result_count(
        context,
        cpu,
        value_map,
        op,
        target_op,
        &symbol,
        &attrs,
        &[oracle],
        COMMITMENT_ARTIFACT_RESULT_TYPES,
        LoweredResultCount::One,
    )
}
