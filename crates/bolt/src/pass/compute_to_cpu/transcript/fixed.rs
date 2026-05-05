use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Cpu};
use crate::mlir::{MeliorContext, MlirError};

use super::super::super::support::{
    compute_to_cpu_op_name, lower_transcript_op, transcript_squeeze_cpu_result_types,
    TranscriptDialect, TranscriptOpFamily, CPU_TRANSCRIPT_STATE_RESULT_TYPES,
};

struct CpuTranscriptDialect;

impl TranscriptDialect for CpuTranscriptDialect {
    type Phase = Cpu;

    const STATE_RESULT_TYPES: &'static [&'static str] = CPU_TRANSCRIPT_STATE_RESULT_TYPES;

    fn classify(source_name: &str) -> Option<TranscriptOpFamily> {
        match source_name {
            "compute.transcript_absorb_bytes" => Some(TranscriptOpFamily::AbsorbBytes),
            "compute.transcript_squeeze" => Some(TranscriptOpFamily::Squeeze),
            _ => None,
        }
    }

    fn target_op_name(operation: OperationRef<'_, '_>) -> String {
        compute_to_cpu_op_name(operation)
    }

    fn squeeze_result_types(
        operation: OperationRef<'_, '_>,
    ) -> Result<[&'static str; 2], MlirError> {
        transcript_squeeze_cpu_result_types(operation)
    }
}

pub(super) fn lower_op<'c, 'a>(
    context: &'c MeliorContext,
    cpu: &'a BoltModule<'c, Cpu>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
) -> Result<bool, MlirError> {
    lower_transcript_op::<CpuTranscriptDialect>(context, cpu, value_map, op)
}
