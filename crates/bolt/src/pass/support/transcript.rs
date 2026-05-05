mod compute;
mod family;
mod notation;
mod result_types;
mod results;

use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Phase};
use crate::mlir::MeliorContext;
use crate::mlir::MlirError;
use crate::schema::operation_name;

pub(in crate::pass) use compute::classify_compute_transcript_op;
pub(in crate::pass) use family::TranscriptOpFamily;
pub(crate) use result_types::transcript_squeeze_protocol_result_type;
pub(in crate::pass) use result_types::{
    transcript_squeeze_compute_result_types, transcript_squeeze_cpu_result_types,
    COMPUTE_TRANSCRIPT_STATE_RESULT_TYPES, CPU_TRANSCRIPT_STATE_RESULT_TYPES,
};
use results::lower_transcript_results;

pub(in crate::pass) trait TranscriptDialect {
    type Phase: Phase;

    const STATE_RESULT_TYPES: &'static [&'static str];

    fn classify(source_name: &str) -> Option<TranscriptOpFamily>;
    fn target_op_name(operation: OperationRef<'_, '_>) -> String;
    fn squeeze_result_types(
        operation: OperationRef<'_, '_>,
    ) -> Result<[&'static str; 2], MlirError>;
}

pub(in crate::pass) fn lower_transcript_op<'c, 'a, D>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, D::Phase>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
) -> Result<bool, MlirError>
where
    D: TranscriptDialect,
{
    let Some(family) = D::classify(operation_name(op).as_str()) else {
        return Ok(false);
    };
    lower_transcript_results::<D>(context, module, value_map, op, family)?;
    Ok(true)
}
