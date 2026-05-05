use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Phase};
use crate::mlir::{MeliorContext, MlirError};

use super::super::super::lowering::append_lowered_result_count;
use super::super::shape::SumcheckProofShape;

pub(super) fn lower_sumcheck_proof_op<'c, 'a, P: Phase>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, P>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
    target_name: &str,
    result_types: &[&str],
    shape: SumcheckProofShape,
) -> Result<(), MlirError> {
    append_lowered_result_count(
        context,
        module,
        value_map,
        op,
        shape.operand_start,
        target_name,
        shape.attrs,
        result_types,
        shape.result_count,
    )
}
