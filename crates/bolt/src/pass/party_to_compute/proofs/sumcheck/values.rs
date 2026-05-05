use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Compute};
use crate::mlir::{MeliorContext, MlirError};
use crate::schema::operation_name;

use super::super::super::super::support::{
    lower_sumcheck_value_op, SumcheckValueDialect, SumcheckValueFamily,
    COMPUTE_SUMCHECK_EVAL_RESULT_TYPES, COMPUTE_SUMCHECK_INSTANCE_RESULT_TYPES,
};

struct PartyToComputeSumcheckValueDialect;

impl SumcheckValueDialect for PartyToComputeSumcheckValueDialect {
    type Phase = Compute;

    const EVAL_RESULT_TYPES: &'static [&'static str] = COMPUTE_SUMCHECK_EVAL_RESULT_TYPES;
    const INSTANCE_RESULT_TYPES: &'static [&'static str] = COMPUTE_SUMCHECK_INSTANCE_RESULT_TYPES;

    fn classify(source_name: &str) -> Option<SumcheckValueFamily> {
        match source_name {
            "piop.sumcheck_eval" => Some(SumcheckValueFamily::Eval),
            "piop.sumcheck_instance_result" => Some(SumcheckValueFamily::InstanceResult),
            _ => None,
        }
    }

    fn target_op_name(operation: OperationRef<'_, '_>) -> String {
        match operation_name(operation).as_str() {
            "piop.sumcheck_eval" => "compute.sumcheck_eval".to_owned(),
            "piop.sumcheck_instance_result" => "compute.sumcheck_instance_result".to_owned(),
            source_name => source_name.to_owned(),
        }
    }
}

pub(super) fn lower_op<'c, 'a>(
    context: &'c MeliorContext,
    compute: &'a BoltModule<'c, Compute>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
) -> Result<bool, MlirError> {
    lower_sumcheck_value_op::<PartyToComputeSumcheckValueDialect>(context, compute, value_map, op)
}
