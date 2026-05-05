use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Cpu};
use crate::mlir::{MeliorContext, MlirError};

use super::super::super::support::{
    append_and_map_result_count, append_lowered_result_count, copy_attrs, required_lowered_operand,
    string_attr, LoweredResultCount,
};

const INIT_ATTRS: &[&str] = &["family", "count"];
const APPEND_ATTRS: &[&str] = &["family", "oracle", "index"];
const RESULT_TYPES: &[&str] = &["!cpu.oracle_family"];

pub(super) fn lower_init<'c, 'a>(
    context: &'c MeliorContext,
    cpu: &'a BoltModule<'c, Cpu>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
) -> Result<(), MlirError> {
    append_lowered_result_count(
        context,
        cpu,
        value_map,
        op,
        0,
        "cpu.oracle_family_init",
        INIT_ATTRS,
        RESULT_TYPES,
        LoweredResultCount::One,
    )
}

pub(super) fn lower_append<'c, 'a>(
    context: &'c MeliorContext,
    cpu: &'a BoltModule<'c, Cpu>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
) -> Result<(), MlirError> {
    let input = required_lowered_operand(
        op,
        value_map,
        0,
        "compute.oracle_family_append input operand was not lowered",
    )?;
    let oracle = required_lowered_operand(
        op,
        value_map,
        1,
        "compute.oracle_family_append oracle operand was not lowered",
    )?;
    let symbol = string_attr(op, "sym_name")?;
    let attrs = copy_attrs(op, APPEND_ATTRS)?;
    append_and_map_result_count(
        context,
        cpu,
        value_map,
        op,
        "cpu.oracle_family_append",
        &symbol,
        &attrs,
        &[input, oracle],
        RESULT_TYPES,
        LoweredResultCount::One,
    )
}
