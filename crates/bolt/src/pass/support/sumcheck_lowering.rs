mod compute;
mod family;
mod notation;
mod result_types;
mod results;

use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Phase};
use crate::mlir::{MeliorContext, MlirError};
use crate::schema::operation_name;

pub(in crate::pass) use compute::classify_compute_sumcheck_value_op;
pub(in crate::pass) use family::SumcheckValueFamily;
pub(in crate::pass) use result_types::{
    COMPUTE_SUMCHECK_EVAL_RESULT_TYPES, COMPUTE_SUMCHECK_INSTANCE_RESULT_TYPES,
    CPU_SUMCHECK_EVAL_RESULT_TYPES, CPU_SUMCHECK_INSTANCE_RESULT_TYPES,
};
use results::lower_sumcheck_results;

pub(in crate::pass) trait SumcheckValueDialect {
    type Phase: Phase;

    const EVAL_RESULT_TYPES: &'static [&'static str];
    const INSTANCE_RESULT_TYPES: &'static [&'static str];

    fn classify(source_name: &str) -> Option<SumcheckValueFamily>;
    fn target_op_name(operation: OperationRef<'_, '_>) -> String;
}

pub(in crate::pass) fn lower_sumcheck_value_op<'c, 'a, D>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, D::Phase>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
) -> Result<bool, MlirError>
where
    D: SumcheckValueDialect,
{
    let Some(family) = D::classify(operation_name(op).as_str()) else {
        return Ok(false);
    };
    lower_sumcheck_results::<D>(
        context,
        module,
        value_map,
        op,
        family.attrs(),
        family.result_types::<D>(),
        family.result_count(),
    )?;
    Ok(true)
}
