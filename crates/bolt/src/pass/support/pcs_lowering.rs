mod compute;
mod family;
mod notation;
mod results;
mod role;

use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Phase};
use crate::mlir::{MeliorContext, MlirError};
use crate::schema::operation_name;

pub(in crate::pass) use compute::classify_compute_pcs_op;
pub(in crate::pass) use family::PcsOpFamily;
use results::lower_pcs_results;
pub(in crate::pass) use role::PcsLoweringRole;

pub(in crate::pass) trait PcsDialect {
    type Phase: Phase;

    const CLAIM_RESULT_TYPES: &'static [&'static str];
    const BATCH_RESULT_TYPES: &'static [&'static str];
    const BATCH_OPENING_RESULT_TYPES: &'static [&'static str];

    fn classify(source_name: &str) -> Option<PcsOpFamily>;
    fn target_op_name(
        operation: OperationRef<'_, '_>,
        role: PcsLoweringRole<'_>,
    ) -> Result<String, MlirError>;
}

pub(in crate::pass) fn lower_pcs_op<'c, 'a, D>(
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, D::Phase>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
    role: PcsLoweringRole<'_>,
) -> Result<bool, MlirError>
where
    D: PcsDialect,
{
    let Some(family) = D::classify(operation_name(op).as_str()) else {
        return Ok(false);
    };
    lower_pcs_results::<D>(
        context,
        module,
        value_map,
        op,
        family.attrs(),
        family.result_types::<D>(),
        family.result_count(),
        role,
    )?;
    Ok(true)
}
