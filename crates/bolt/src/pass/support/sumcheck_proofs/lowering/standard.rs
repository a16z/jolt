use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Phase};
use crate::mlir::{MeliorContext, MlirError};

use super::super::shape::{sumcheck_batch_shape, SUMCHECK_CLAIM_SHAPE, SUMCHECK_DRIVER_SHAPE};
use super::core::lower_sumcheck_proof_op;

pub(in crate::pass) fn lower_sumcheck_claim_op<'c, 'a, P: Phase>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, P>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
    target_name: &str,
    result_types: &[&str],
) -> Result<(), MlirError> {
    lower_sumcheck_proof_op(
        context,
        module,
        value_map,
        op,
        target_name,
        result_types,
        SUMCHECK_CLAIM_SHAPE,
    )
}

pub(in crate::pass) fn lower_sumcheck_batch_op<'c, 'a, P: Phase>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, P>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
    operand_start: usize,
    target_name: &str,
    result_types: &[&str],
) -> Result<(), MlirError> {
    lower_sumcheck_proof_op(
        context,
        module,
        value_map,
        op,
        target_name,
        result_types,
        sumcheck_batch_shape(operand_start),
    )
}

pub(in crate::pass) fn lower_sumcheck_driver_op<'c, 'a, P: Phase>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, P>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
    target_name: &str,
    result_types: &[&str],
) -> Result<(), MlirError> {
    lower_sumcheck_proof_op(
        context,
        module,
        value_map,
        op,
        target_name,
        result_types,
        SUMCHECK_DRIVER_SHAPE,
    )
}
