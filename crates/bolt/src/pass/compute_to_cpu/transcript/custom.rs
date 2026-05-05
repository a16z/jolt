use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Cpu};
use crate::mlir::{MeliorContext, MlirError};

use super::attrs::{transcript_absorb_attrs, transcript_init_attrs};
use crate::pass::support::{
    append_and_map_result_count, required_lowered_operand, string_attr, LoweredResultCount,
    CPU_TRANSCRIPT_STATE_RESULT_TYPES,
};

pub(super) fn lower_init<'c, 'a>(
    context: &'c MeliorContext,
    cpu: &'a BoltModule<'c, Cpu>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
) -> Result<(), MlirError> {
    let attrs = transcript_init_attrs(op)?;
    let symbol = string_attr(op, "sym_name")?;
    append_and_map_result_count(
        context,
        cpu,
        value_map,
        op,
        "cpu.transcript_init",
        &symbol,
        &attrs,
        &[],
        CPU_TRANSCRIPT_STATE_RESULT_TYPES,
        LoweredResultCount::One,
    )
}

pub(super) fn lower_absorb<'c, 'a>(
    context: &'c MeliorContext,
    cpu: &'a BoltModule<'c, Cpu>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
) -> Result<(), MlirError> {
    let input = required_lowered_operand(
        op,
        value_map,
        0,
        "compute.transcript_absorb input operand was not lowered",
    )?;
    let artifact = required_lowered_operand(
        op,
        value_map,
        1,
        "compute.transcript_absorb artifact operand was not lowered",
    )?;
    let attrs = transcript_absorb_attrs(op)?;
    let symbol = string_attr(op, "sym_name")?;
    append_and_map_result_count(
        context,
        cpu,
        value_map,
        op,
        "cpu.transcript_absorb",
        &symbol,
        &attrs,
        &[input, artifact],
        CPU_TRANSCRIPT_STATE_RESULT_TYPES,
        LoweredResultCount::One,
    )
}
