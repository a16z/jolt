mod dialect;

use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Compute};
use crate::mlir::{MeliorContext, MlirError};

use super::super::support::lower_transcript_op;

pub(super) fn lower_op<'c, 'a>(
    context: &'c MeliorContext,
    kernelized: &'a BoltModule<'c, Compute>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
) -> Result<bool, MlirError> {
    lower_transcript_op::<dialect::KernelResolutionTranscriptDialect>(
        context, kernelized, value_map, op,
    )
}
