use crate::ir::{BoltModule, Compute, Party};
use crate::mlir::{MeliorContext, MlirError};
use crate::pass::PartyToComputeLowering;

pub(super) use crate::pass::{string_attr, transcript_squeeze_protocol_result_type};

pub(super) fn lower_party_to_compute<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Party>,
    function_symbol: &str,
    source_symbol: &str,
    stage_label: &str,
) -> Result<BoltModule<'c, Compute>, MlirError> {
    crate::pass::lower_party_to_compute(
        context,
        module,
        PartyToComputeLowering {
            params_symbol: "jolt.compute_params",
            function_symbol,
            source_symbol,
            diagnostic_label: stage_label,
        },
    )
}
