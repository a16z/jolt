mod dialect;

use std::collections::BTreeMap;

use melior::ir::operation::OperationRef;
use melior::ir::Value;

use crate::ir::{BoltModule, Compute, Role};
use crate::mlir::{MeliorContext, MlirError};

use super::super::super::support::{lower_pcs_op, PcsLoweringRole};

pub(super) fn lower_op<'c, 'a>(
    context: &'c MeliorContext,
    compute: &'a BoltModule<'c, Compute>,
    value_map: &mut BTreeMap<String, Value<'c, 'a>>,
    op: OperationRef<'_, '_>,
    role: &Role,
) -> Result<bool, MlirError> {
    lower_pcs_op::<dialect::PartyToComputePcsDialect>(
        context,
        compute,
        value_map,
        op,
        PcsLoweringRole::available(role),
    )
}
