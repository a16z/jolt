use melior::ir::operation::OperationRef;

use crate::ir::Compute;
use crate::mlir::MlirError;
use crate::schema::operation_name;

use super::super::super::support::{
    classify_compute_transcript_op, transcript_squeeze_compute_result_types, TranscriptDialect,
    TranscriptOpFamily, COMPUTE_TRANSCRIPT_STATE_RESULT_TYPES,
};

pub(super) struct KernelResolutionTranscriptDialect;

impl TranscriptDialect for KernelResolutionTranscriptDialect {
    type Phase = Compute;

    const STATE_RESULT_TYPES: &'static [&'static str] = COMPUTE_TRANSCRIPT_STATE_RESULT_TYPES;

    fn classify(source_name: &str) -> Option<TranscriptOpFamily> {
        classify_compute_transcript_op(source_name)
    }

    fn target_op_name(operation: OperationRef<'_, '_>) -> String {
        operation_name(operation)
    }

    fn squeeze_result_types(
        operation: OperationRef<'_, '_>,
    ) -> Result<[&'static str; 2], MlirError> {
        transcript_squeeze_compute_result_types(operation)
    }
}
