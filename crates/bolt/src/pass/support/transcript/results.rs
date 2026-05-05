use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::BoltModule;
use crate::mlir::{MeliorContext, MlirError};

use super::super::lowering::append_lowered_result_count;
use super::{TranscriptDialect, TranscriptOpFamily};

pub(super) fn lower_transcript_results<'c, 'a, D>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, D::Phase>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
    family: TranscriptOpFamily,
) -> Result<(), MlirError>
where
    D: TranscriptDialect,
{
    let target_name = D::target_op_name(op);
    let result_types = family.result_types::<D>(op)?;
    append_lowered_result_count(
        context,
        module,
        value_map,
        op,
        0,
        &target_name,
        family.attrs(),
        result_types.as_slice(),
        family.result_count(),
    )
}
