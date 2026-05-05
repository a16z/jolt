use melior::ir::operation::OperationRef;

use crate::ir::Compute;
use crate::mlir::MlirError;
use crate::schema::operation_name;

use super::super::super::support::{
    transcript_squeeze_compute_result_types, TranscriptDialect, TranscriptOpFamily,
    COMPUTE_TRANSCRIPT_STATE_RESULT_TYPES,
};

pub(super) struct PartyToComputeTranscriptDialect;

impl TranscriptDialect for PartyToComputeTranscriptDialect {
    type Phase = Compute;

    const STATE_RESULT_TYPES: &'static [&'static str] = COMPUTE_TRANSCRIPT_STATE_RESULT_TYPES;

    fn classify(source_name: &str) -> Option<TranscriptOpFamily> {
        match source_name {
            "transcript.state" => Some(TranscriptOpFamily::Init),
            "transcript.absorb_bytes" => Some(TranscriptOpFamily::AbsorbBytes),
            "transcript.squeeze" => Some(TranscriptOpFamily::Squeeze),
            _ => None,
        }
    }

    fn target_op_name(operation: OperationRef<'_, '_>) -> String {
        match operation_name(operation).as_str() {
            "transcript.state" => "compute.transcript_init".to_owned(),
            "transcript.absorb_bytes" => "compute.transcript_absorb_bytes".to_owned(),
            "transcript.squeeze" => "compute.transcript_squeeze".to_owned(),
            source_name => source_name.to_owned(),
        }
    }

    fn squeeze_result_types(
        operation: OperationRef<'_, '_>,
    ) -> Result<[&'static str; 2], MlirError> {
        transcript_squeeze_compute_result_types(operation)
    }
}
