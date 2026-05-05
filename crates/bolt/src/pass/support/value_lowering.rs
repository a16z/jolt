mod compute;
mod expressions;
mod family;
mod fixed;
mod notation;
mod result_types;

use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Phase};
use crate::mlir::{MeliorContext, MlirError};
use crate::schema::operation_name;

pub(in crate::pass) use compute::classify_compute_value_op;
use expressions::lower_field_expression;
pub(in crate::pass) use family::ValueOpFamily;
use fixed::lower_fixed_results;
pub(in crate::pass) use result_types::{
    COMPUTE_FIELD_RESULT_TYPES, COMPUTE_OPENING_INPUT_RESULT_TYPES, COMPUTE_POINT_RESULT_TYPES,
    CPU_FIELD_RESULT_TYPES, CPU_OPENING_INPUT_RESULT_TYPES, CPU_POINT_RESULT_TYPES,
};

pub(in crate::pass) trait ValueDialect {
    type Phase: Phase;

    const OPENING_INPUT_RESULT_TYPES: &'static [&'static str];
    const POINT_RESULT_TYPES: &'static [&'static str];
    const FIELD_RESULT_TYPES: &'static [&'static str];

    fn classify(source_name: &str) -> Option<ValueOpFamily>;
    fn target_op_name(operation: OperationRef<'_, '_>) -> String;
}

pub(in crate::pass) fn lower_value_op<'c, 'a, D>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, D::Phase>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
) -> Result<bool, MlirError>
where
    D: ValueDialect,
{
    let Some(family) = D::classify(operation_name(op).as_str()) else {
        return Ok(false);
    };
    match family.fixed_shape::<D>() {
        Some(shape) => {
            lower_fixed_results::<D>(
                context,
                module,
                value_map,
                op,
                shape.attrs,
                shape.result_types,
                shape.result_count,
            )?;
        }
        None => lower_field_expression::<D>(context, module, value_map, op)?,
    }
    Ok(true)
}
